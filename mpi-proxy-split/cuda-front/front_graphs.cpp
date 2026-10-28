/****************************************************************************
 *  Copyright (C) 2026 by Yao Xu                                            *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

// CUDA graphs and stream capture (cuda/DESIGN-lower-half-cuda.md, Phase 3),
// in the runtime API.  Graphs, their nodes and executable graphs are
// records; a node gets one when the application first sees it.
//
//  * A graph is rebuilt from what it is at the checkpoint: the front reads
//    each live graph back node by node (types, parameters, edges), whether
//    the application built it or a capture did.  A restart adds the nodes
//    in the same order, then all edges; a child graph node is made from a
//    graph built the same way.  Kernel arguments are copied by the sizes of
//    cuFuncGetParamInfo.  The node handles that the application holds stay
//    valid.
//  * An executable graph is instantiated from a clone of the front's: its
//    nodes, which the driver takes in node updates, are the clone's, which
//    the front can rebuild.  The record keeps that clone, a clone of the
//    graph of the last cudaGraphExecUpdate (which sets every node), the
//    node updates since (cudaGraphExec*NodeSetParams), and the nodes'
//    enabled states.  A restart instantiates it again in that order.
//  * A stream capture holds the checkpoint gate from its beginning to its
//    end: the checkpoint's synchronization would end it.  It must end on
//    the thread that began it.
//
// Host nodes run the application's function through a target of the
// front's (front_async.cpp), which lives as long as the process.
//
// Not supported: memory nodes (graph memory), conditional nodes, external
// semaphore nodes, user objects, device launch and device-updatable kernel
// nodes (handles inside kernels), kernel nodes with events in their
// attributes, the generic cudaGraphAddNode and cudaGraph*NodeSetParams,
// kernels of the libraries (cuBLAS, ...) in graphs, and the driver API's
// graph functions.

#include <string.h>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "front_objs.h"

int front_captures;

namespace {

struct Graph;
struct GraphSnap;

struct GNode : FrontObj {
  Graph *graph;
  Graph *child;          // a child graph node's own graph, once a record
};

enum Owner {
  APP,                   // the application's
  CLONE,                 // the front's, for an executable graph
  EMBEDDED               // a child graph node's
};

struct Graph : FrontObj {
  Owner owner;
  Ctx *ctx;              // current at its making: a restart builds it there
  std::vector<GNode *> nodes;  // those with records
  GraphSnap *snap;       // during a checkpoint (not EMBEDDED)
};

// A node as read back, or as the application set it in an executable
// graph.  Handles are records.
struct NodeSnap {
  cudaGraphNodeType type;
  GNode *node;
  // Kernel.
  bool driver;           // a function of cuModuleGetFunction
  void *func;            // the host stub, or the function's record
  Ctx *ctx;
  dim3 grid, block;
  unsigned shared;
  std::vector<char> args;
  std::vector<size_t> offsets;
  std::vector<std::pair<int, cudaLaunchAttributeValue> > attrs;
  // The others.
  cudaMemcpy3DParms copy;    // arrays as records
  cudaMemsetParams set;
  void *host_fn;             // the driver's (front_host_node())
  void *host_data;
  FrontObj *event;
  std::unique_ptr<GraphSnap> child;  // read back
  Graph *child_clone;        // set in an executable graph
};

struct GraphSnap {
  std::vector<NodeSnap> nodes;
  std::vector<int> from, to;
  std::vector<cudaGraphEdgeData> edges;
};

struct Exec : FrontObj {
  Ctx *ctx;
  bool gone;             // its context was destroyed
  unsigned long long flags;
  Graph *shadow;         // the clone that it was instantiated from
  Graph *update;         // a clone of the last cudaGraphExecUpdate's graph
  std::unordered_map<GNode *, GNode *> of_template;  // application -> shadow
  std::map<GNode *, NodeSnap> overrides;  // by shadow node, since 'update'
  std::map<GNode *, unsigned> enabled;    // by shadow node
  std::vector<FrontObj *> records;        // events that its nodes record
};

// Of this generation's real handles; with objs_lock.
std::unordered_map<void *, Graph *> graph_by_real;
std::unordered_map<void *, GNode *> node_by_real;
// Captures in progress, by the stream that began them: the thread that
// holds the gate, and the capture's id.  And the records that
// cudaStreamGetCaptureInfo made for the graphs of captures, by id: a
// capture that fails takes its graph with it.
struct Capture {
  FrontThread *thread;
  unsigned long long id;
};
std::map<cudaStream_t, Capture> capturing;
std::map<unsigned long long, Graph *> capture_graphs;

// ---- Records -------------------------------------------------------------

// With objs_lock held.
Graph *
graph_rec(cudaGraph_t real, Owner owner, Ctx *ctx = NULL)
{
  if (real == NULL) {
    return NULL;
  }
  auto it = graph_by_real.find(real);
  if (it != graph_by_real.end()) {
    return it->second;
  }
  Graph *g = new Graph();
  g->real = real;
  g->owner = owner;
  g->ctx = ctx;
  obj_link(g, FRONT_GRAPH);
  graph_by_real[real] = g;
  return g;
}

GNode *
node_rec(cudaGraphNode_t real, Graph *g)
{
  if (real == NULL) {
    return NULL;
  }
  auto it = node_by_real.find(real);
  if (it != node_by_real.end()) {
    return it->second;
  }
  GNode *n = new GNode();
  n->real = real;
  n->graph = g;
  obj_link(n, FRONT_GNODE);
  node_by_real[real] = n;
  if (g != NULL) {
    g->nodes.push_back(n);
  }
  return n;
}

void forget_graph(Graph *g);

void
forget_node(GNode *n, bool from_graph)
{
  if (n->child != NULL) {
    forget_graph(n->child);
  }
  for (FrontObj *o = obj_heads[FRONT_GEXEC]; o != NULL; o = o->next) {
    ((Exec *)o)->of_template.erase(n);
  }
  if (from_graph && n->graph != NULL) {
    auto &v = n->graph->nodes;
    v.erase(std::remove(v.begin(), v.end(), n), v.end());
  }
  node_by_real.erase(n->real);
  obj_unlink(n);
  delete n;
}

void
forget_graph(Graph *g)
{
  for (GNode *n : g->nodes) {
    forget_node(n, false);
  }
  delete g->snap;
  graph_by_real.erase(g->real);
  obj_unlink(g);
  delete g;
}

Graph *
graph_of(cudaGraph_t graph)
{
  front_real_obj(graph, FRONT_GRAPH);
  return (Graph *)graph;
}

GNode *
node_of(cudaGraphNode_t node)
{
  front_real_obj(node, FRONT_GNODE);
  return (GNode *)node;
}

Exec *
exec_of(cudaGraphExec_t exec)
{
  front_real_obj(exec, FRONT_GEXEC);
  return (Exec *)exec;
}

cudaGraph_t
real_graph(cudaGraph_t graph)
{
  return (cudaGraph_t)front_real_obj(graph, FRONT_GRAPH);
}

cudaGraphNode_t
real_node(cudaGraphNode_t node)
{
  return (cudaGraphNode_t)front_real_obj(node, FRONT_GNODE);
}

std::vector<cudaGraphNode_t>
real_nodes(const cudaGraphNode_t *nodes, size_t n)
{
  std::vector<cudaGraphNode_t> v;
  for (size_t i = 0; nodes != NULL && i < n; i++) {
    v.push_back(real_node(nodes[i]));
  }
  return v;
}

// The records of 'n' real nodes of 'g', in place.
void
virtual_nodes(cudaGraphNode_t *nodes, size_t n, Graph *g)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  for (size_t i = 0; nodes != NULL && i < n; i++) {
    nodes[i] = (cudaGraphNode_t)node_rec(nodes[i], g);
  }
}

// The record of kind 'kind' whose real handle is 'real', or NULL; with
// objs_lock held.
FrontObj *
record_of(int kind, void *real)
{
  for (FrontObj *o = obj_heads[kind]; o != NULL && real != NULL;
       o = o->next) {
    if (o->real == real) {
      return o;
    }
  }
  return NULL;
}

// The arrays of a copy, from records to real ones or back.
void
copy_to_real(cudaMemcpy3DParms *p)
{
  p->srcArray = (cudaArray_t)front_real_obj(p->srcArray, FRONT_ARRAY);
  p->dstArray = (cudaArray_t)front_real_obj(p->dstArray, FRONT_ARRAY);
}

void
copy_to_records(cudaMemcpy3DParms *p)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  for (cudaArray_t *a : { &p->srcArray, &p->dstArray }) {
    if (*a != NULL) {
      FrontObj *o = record_of(FRONT_ARRAY, *a);
      if (o == NULL) {
        front_fatal("a graph's copy names array %p, which the application "
                    "did not make\n", (void *)*a);
      }
      *a = (cudaArray_t)o;
    }
  }
}

FrontObj *
event_record(cudaEvent_t real)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  FrontObj *o = record_of(FRONT_EVENT, real);
  if (o == NULL && real != NULL) {
    front_fatal("a graph's node names event %p, which the application did "
                "not make\n", (void *)real);
  }
  return o;
}

// ---- Nodes, read back ----------------------------------------------------

// A kernel's arguments, copied by the offsets and sizes of its parameters:
// from an array of pointers, or a buffer of cuLaunchKernel's 'extra'.
void
copy_args(CUfunction f, void **params, void **extra, NodeSnap *s)
{
  std::vector<std::pair<size_t, size_t> > info;
  for (int i = 0;; i++) {
    size_t offset = 0, size = 0;
    CUresult rc;
    FRONT_LH(rc = DRV(cuFuncGetParamInfo)(f, i, &offset, &size));
    if (rc != CUDA_SUCCESS) {
      break;
    }
    info.emplace_back(offset, size);
  }
  size_t total = info.empty() ? 0 : info.back().first + info.back().second;
  s->args.assign(total, 0);
  s->offsets.clear();
  const char *buffer = NULL;
  size_t buffer_size = 0;
  for (size_t i = 0; params == NULL && extra != NULL &&
                     extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
    if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      buffer = (const char *)extra[i + 1];
    } else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
      buffer_size = *(size_t *)extra[i + 1];
    }
  }
  for (size_t i = 0; i < info.size(); i++) {
    s->offsets.push_back(info[i].first);
    if (params != NULL) {
      memcpy(&s->args[info[i].first], params[i], info[i].second);
    } else if (buffer != NULL &&
               info[i].first + info[i].second <= buffer_size) {
      memcpy(&s->args[info[i].first], buffer + info[i].first,
             info[i].second);
    }
  }
}

// The attributes that a kernel node may have, but those that hold events
// (7, 12).
const int kernel_attrs[] = {
  cudaLaunchAttributeAccessPolicyWindow,
  cudaLaunchAttributeCooperative,
  cudaLaunchAttributeSynchronizationPolicy,
  cudaLaunchAttributeClusterDimension,
  cudaLaunchAttributeClusterSchedulingPolicyPreference,
  cudaLaunchAttributeProgrammaticStreamSerialization,
  cudaLaunchAttributePriority,
  cudaLaunchAttributeMemSyncDomainMap,
  cudaLaunchAttributeMemSyncDomain,
  cudaLaunchAttributePreferredClusterDimension,
  cudaLaunchAttributeDeviceUpdatableKernelNode,
  cudaLaunchAttributePreferredSharedMemoryCarveout,
  cudaLaunchAttributeNvlinkUtilCentricScheduling,
};

void
snap_kernel(cudaGraphNode_t real, NodeSnap *s)
{
  CUDA_KERNEL_NODE_PARAMS d;
  CUresult rc;
  FRONT_LH(rc = DRV(cuGraphKernelNodeGetParams_v2)((CUgraphNode)real, &d));
  if (rc != CUDA_SUCCESS) {
    front_fatal("reading a graph's kernel node: error %d\n", rc);
  }
  s->grid = dim3(d.gridDimX, d.gridDimY, d.gridDimZ);
  s->block = dim3(d.blockDimX, d.blockDimY, d.blockDimZ);
  s->shared = d.sharedMemBytes;
  s->ctx = front_ctx_of_real(d.ctx);
  cudaKernelNodeParams p;
  cudaError_t err;
  FRONT_LH(err = REAL(cudaGraphKernelNodeGetParams)(real, &p));
  bool registered = false;
  if (err == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    registered = front_registered_kernel(p.func);
  } else {
    FRONT_LH(err = REAL(cudaGetLastError)());
  }
  if (registered) {
    s->driver = false;
    s->func = p.func;
  } else {
    std::lock_guard<std::mutex> guard(objs_lock);
    s->driver = true;
    s->func = record_of(FRONT_FUNCTION, d.func);
    if (s->func == NULL) {
      front_fatal("a graph's kernel node runs a kernel that the application "
                  "neither registered nor loaded: one of a library's "
                  "(cuBLAS, cuFFT, ...)?\n");
    }
  }
  copy_args(d.func, d.kernelParams, d.extra, s);
  for (int id : kernel_attrs) {
    cudaLaunchAttributeValue v;
    memset(&v, 0, sizeof v);
    FRONT_LH(err = REAL(cudaGraphKernelNodeGetAttribute)(
               real, (cudaLaunchAttributeID)id, &v));
    if (err != cudaSuccess) {
      FRONT_LH(err = REAL(cudaGetLastError)());
      continue;
    }
    if (id == cudaLaunchAttributeDeviceUpdatableKernelNode &&
        v.deviceUpdatableKernelNode.deviceUpdatable) {
      front_fatal("device-updatable kernel nodes are not supported\n");
    }
    s->attrs.emplace_back(id, v);
  }
}

void snap_graph(Graph *g, GraphSnap *out);

void
snap_node(cudaGraphNode_t real, Graph *g, NodeSnap *s)
{
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    s->node = node_rec(real, g);
  }
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphNodeGetType)(real, &s->type));
  switch (s->type) {
    case cudaGraphNodeTypeKernel:
      snap_kernel(real, s);
      break;
    case cudaGraphNodeTypeMemcpy:
      FRONT_LH(rc = REAL(cudaGraphMemcpyNodeGetParams)(real, &s->copy));
      copy_to_records(&s->copy);
      break;
    case cudaGraphNodeTypeMemset:
      FRONT_LH(rc = REAL(cudaGraphMemsetNodeGetParams)(real, &s->set));
      break;
    case cudaGraphNodeTypeHost: {
      cudaHostNodeParams p;
      FRONT_LH(rc = REAL(cudaGraphHostNodeGetParams)(real, &p));
      cudaHostFn_t fn;
      void *data;
      if (rc == cudaSuccess &&
          !front_host_node_target((void *)p.fn, p.userData, &fn, &data)) {
        front_fatal("a graph's host node that the front did not make\n");
      }
      s->host_fn = (void *)p.fn;
      s->host_data = p.userData;
      break;
    }
    case cudaGraphNodeTypeEmpty:
      break;
    case cudaGraphNodeTypeWaitEvent:
    case cudaGraphNodeTypeEventRecord: {
      cudaEvent_t e = NULL;
      if (s->type == cudaGraphNodeTypeWaitEvent) {
        FRONT_LH(rc = REAL(cudaGraphEventWaitNodeGetEvent)(real, &e));
      } else {
        FRONT_LH(rc = REAL(cudaGraphEventRecordNodeGetEvent)(real, &e));
      }
      s->event = event_record(e);
      break;
    }
    case cudaGraphNodeTypeGraph: {
      cudaGraph_t child = NULL;
      FRONT_LH(rc = REAL(cudaGraphChildGraphNodeGetGraph)(real, &child));
      Graph *c;
      {
        std::lock_guard<std::mutex> guard(objs_lock);
        c = s->node->child = graph_rec(child, EMBEDDED, g->ctx);
      }
      s->child.reset(new GraphSnap());
      snap_graph(c, s->child.get());
      break;
    }
    default:
      front_fatal("graph nodes of type %d (memory, conditional, external "
                  "semaphore, ...) are not supported\n", (int)s->type);
  }
  if (rc != cudaSuccess) {
    front_fatal("reading a graph's node of type %d: error %d\n",
                (int)s->type, rc);
  }
}

void
snap_graph(Graph *g, GraphSnap *out)
{
  cudaGraph_t real = (cudaGraph_t)g->real;
  size_t n = 0;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphGetNodes)(real, NULL, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  if (rc == cudaSuccess && n > 0) {
    FRONT_LH(rc = REAL(cudaGraphGetNodes)(real, nodes.data(), &n));
  }
  size_t e = 0;
  if (rc == cudaSuccess) {
    FRONT_LH(rc = REAL(cudaGraphGetEdges)(real, NULL, NULL, NULL, &e));
  }
  std::vector<cudaGraphNode_t> from(e), to(e);
  out->edges.resize(e);
  if (rc == cudaSuccess && e > 0) {
    FRONT_LH(rc = REAL(cudaGraphGetEdges)(real, from.data(), to.data(),
                                          out->edges.data(), &e));
  }
  if (rc != cudaSuccess) {
    front_fatal("reading a graph: error %d\n", rc);
  }
  std::unordered_map<cudaGraphNode_t, int> index;
  out->nodes.resize(n);
  for (size_t i = 0; i < n; i++) {
    index[nodes[i]] = (int)i;
    snap_node(nodes[i], g, &out->nodes[i]);
  }
  for (size_t i = 0; i < e; i++) {
    out->from.push_back(index.at(from[i]));
    out->to.push_back(index.at(to[i]));
  }
}

// ---- Nodes, made again ---------------------------------------------------

cudaError_t
add_kernel(cudaGraph_t g, NodeSnap &s, cudaGraphNode_t *n, Ctx *graph_ctx)
{
  std::vector<void *> params;
  for (size_t offset : s.offsets) {
    params.push_back(&s.args[offset]);
  }
  if (s.ctx != NULL && s.ctx != graph_ctx) {
    front_ctx_use(s.ctx);
  }
  cudaError_t rc;
  if (!s.driver) {
    cudaKernelNodeParams p = cudaKernelNodeParams();
    p.func = s.func;
    p.gridDim = s.grid;
    p.blockDim = s.block;
    p.sharedMemBytes = s.shared;
    p.kernelParams = params.empty() ? NULL : params.data();
    FRONT_LH(rc = REAL(cudaGraphAddKernelNode)(n, g, NULL, 0, &p));
  } else {
    CUDA_KERNEL_NODE_PARAMS p = CUDA_KERNEL_NODE_PARAMS();
    p.func = (CUfunction)((FrontObj *)s.func)->real;
    p.gridDimX = s.grid.x;
    p.gridDimY = s.grid.y;
    p.gridDimZ = s.grid.z;
    p.blockDimX = s.block.x;
    p.blockDimY = s.block.y;
    p.blockDimZ = s.block.z;
    p.sharedMemBytes = s.shared;
    p.kernelParams = params.empty() ? NULL : params.data();
    CUresult drc;
    FRONT_LH(drc = DRV(cuGraphAddKernelNode_v2)((CUgraphNode *)n, (CUgraph)g,
                                                NULL, 0, &p));
    rc = (cudaError_t)drc;
  }
  for (auto &attr : s.attrs) {
    if (rc == cudaSuccess) {
      FRONT_LH(rc = REAL(cudaGraphKernelNodeSetAttribute)(
                 *n, (cudaLaunchAttributeID)attr.first, &attr.second));
    }
  }
  if (s.ctx != NULL && s.ctx != graph_ctx && graph_ctx != NULL) {
    front_ctx_use(graph_ctx);
  }
  return rc;
}

void map_embedded(GraphSnap &snap, Graph *rec, cudaGraph_t real);

// Adds the nodes of 'snap' to 'g', then its edges.
void
build(GraphSnap &snap, cudaGraph_t g, Ctx *ctx)
{
  std::vector<cudaGraphNode_t> made;
  for (NodeSnap &s : snap.nodes) {
    cudaGraphNode_t n = NULL;
    cudaError_t rc = cudaSuccess;
    switch (s.type) {
      case cudaGraphNodeTypeKernel:
        rc = add_kernel(g, s, &n, ctx);
        break;
      case cudaGraphNodeTypeMemcpy: {
        cudaMemcpy3DParms p = s.copy;
        copy_to_real(&p);
        FRONT_LH(rc = REAL(cudaGraphAddMemcpyNode)(&n, g, NULL, 0, &p));
        break;
      }
      case cudaGraphNodeTypeMemset:
        FRONT_LH(rc = REAL(cudaGraphAddMemsetNode)(&n, g, NULL, 0, &s.set));
        break;
      case cudaGraphNodeTypeHost: {
        cudaHostNodeParams p = { (cudaHostFn_t)s.host_fn, s.host_data };
        FRONT_LH(rc = REAL(cudaGraphAddHostNode)(&n, g, NULL, 0, &p));
        break;
      }
      case cudaGraphNodeTypeEmpty:
        FRONT_LH(rc = REAL(cudaGraphAddEmptyNode)(&n, g, NULL, 0));
        break;
      case cudaGraphNodeTypeWaitEvent:
        FRONT_LH(rc = REAL(cudaGraphAddEventWaitNode)(
                   &n, g, NULL, 0, (cudaEvent_t)s.event->real));
        break;
      case cudaGraphNodeTypeEventRecord:
        FRONT_LH(rc = REAL(cudaGraphAddEventRecordNode)(
                   &n, g, NULL, 0, (cudaEvent_t)s.event->real));
        break;
      case cudaGraphNodeTypeGraph: {
        cudaGraph_t temp = NULL;
        FRONT_LH(rc = REAL(cudaGraphCreate)(&temp, 0));
        if (rc == cudaSuccess) {
          build(*s.child, temp, ctx);
          FRONT_LH(rc = REAL(cudaGraphAddChildGraphNode)(&n, g, NULL, 0,
                                                         temp));
          cudaError_t destroyed;
          FRONT_LH(destroyed = REAL(cudaGraphDestroy)(temp));
          (void)destroyed;
        }
        break;
      }
      default:
        break;
    }
    if (rc != cudaSuccess) {
      front_fatal("making a graph's node of type %d again: error %d\n",
                  (int)s.type, rc);
    }
    s.node->real = n;
    made.push_back(n);
  }
  std::vector<cudaGraphNode_t> from, to;
  for (size_t i = 0; i < snap.from.size(); i++) {
    from.push_back(made[snap.from[i]]);
    to.push_back(made[snap.to[i]]);
  }
  cudaError_t rc = cudaSuccess;
  if (!from.empty()) {
    FRONT_LH(rc = REAL(cudaGraphAddDependencies)(g, from.data(), to.data(),
                                                 snap.edges.data(),
                                                 from.size()));
  }
  if (rc != cudaSuccess) {
    front_fatal("making a graph's edges again: error %d\n", rc);
  }
  for (size_t i = 0; i < snap.nodes.size(); i++) {
    NodeSnap &s = snap.nodes[i];
    if (s.type == cudaGraphNodeTypeGraph) {
      cudaGraph_t embedded = NULL;
      FRONT_LH(rc = REAL(cudaGraphChildGraphNodeGetGraph)(made[i],
                                                          &embedded));
      map_embedded(*s.child, s.node->child, embedded);
    }
  }
}

// A child graph node's own graph is a copy of the graph that it was made
// from, with the nodes in the same order (a probe): so are the records.
void
map_embedded(GraphSnap &snap, Graph *rec, cudaGraph_t real)
{
  if (rec != NULL) {
    rec->real = real;
  }
  size_t n = snap.nodes.size();
  std::vector<cudaGraphNode_t> nodes(n);
  cudaError_t rc = cudaSuccess;
  if (n > 0) {
    FRONT_LH(rc = REAL(cudaGraphGetNodes)(real, nodes.data(), &n));
  }
  if (rc != cudaSuccess || n != snap.nodes.size()) {
    front_fatal("a child graph node's graph has %zu nodes, not %zu\n", n,
                snap.nodes.size());
  }
  for (size_t i = 0; i < n; i++) {
    NodeSnap &s = snap.nodes[i];
    s.node->real = nodes[i];
    if (s.type == cudaGraphNodeTypeGraph) {
      cudaGraph_t embedded = NULL;
      FRONT_LH(rc = REAL(cudaGraphChildGraphNodeGetGraph)(nodes[i],
                                                          &embedded));
      map_embedded(*s.child, s.node->child, embedded);
    }
  }
}

// ---- Executable graphs ---------------------------------------------------

// The events that the event record nodes of 'real' record, and of its child
// graphs.
void
collect_records(cudaGraph_t real, std::vector<FrontObj *> *out)
{
  size_t n = 0;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphGetNodes)(real, NULL, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  if (rc == cudaSuccess && n > 0) {
    FRONT_LH(rc = REAL(cudaGraphGetNodes)(real, nodes.data(), &n));
  }
  for (size_t i = 0; rc == cudaSuccess && i < n; i++) {
    cudaGraphNodeType type;
    FRONT_LH(rc = REAL(cudaGraphNodeGetType)(nodes[i], &type));
    if (type == cudaGraphNodeTypeEventRecord) {
      cudaEvent_t e = NULL;
      FRONT_LH(rc = REAL(cudaGraphEventRecordNodeGetEvent)(nodes[i], &e));
      FrontObj *o = event_record(e);
      if (o != NULL) {
        out->push_back(o);
      }
    } else if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child = NULL;
      FRONT_LH(rc = REAL(cudaGraphChildGraphNodeGetGraph)(nodes[i], &child));
      collect_records(child, out);
    }
  }
}

// A clone of 'graph' for the front, with its record.
cudaError_t
clone_for_front(cudaGraph_t real, Graph **out)
{
  cudaGraph_t clone = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphClone)(&clone, real));
  if (rc == cudaSuccess) {
    Ctx *ctx = front_object_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    *out = graph_rec(clone, CLONE, ctx);
  }
  return rc;
}

void
destroy_clone(Graph *g)
{
  if (g == NULL) {
    return;
  }
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphDestroy)((cudaGraph_t)g->real));
  (void)rc;
  std::lock_guard<std::mutex> guard(objs_lock);
  forget_graph(g);
}

void
drop_overrides(Exec *e)
{
  for (auto &entry : e->overrides) {
    destroy_clone(entry.second.child_clone);
  }
  e->overrides.clear();
}

// The shadow's node for the application's node 'node' of the graph that
// 'e' was instantiated from, or NULL.
GNode *
shadow_node(Exec *e, cudaGraphNode_t node)
{
  GNode *n = node_of(node);
  std::lock_guard<std::mutex> guard(objs_lock);
  auto it = e->of_template.find(n);
  return it != e->of_template.end() ? it->second : NULL;
}

// Applies a node update of 'e' to the real exec.
cudaError_t
apply_override(cudaGraphExec_t real, cudaGraphNode_t node, NodeSnap &s)
{
  cudaError_t rc = cudaSuccess;
  switch (s.type) {
    case cudaGraphNodeTypeKernel: {
      std::vector<void *> params;
      for (size_t offset : s.offsets) {
        params.push_back(&s.args[offset]);
      }
      cudaKernelNodeParams p = cudaKernelNodeParams();
      p.func = s.func;
      p.gridDim = s.grid;
      p.blockDim = s.block;
      p.sharedMemBytes = s.shared;
      p.kernelParams = params.empty() ? NULL : params.data();
      FRONT_LH(rc = REAL(cudaGraphExecKernelNodeSetParams)(real, node, &p));
      break;
    }
    case cudaGraphNodeTypeMemcpy: {
      cudaMemcpy3DParms p = s.copy;
      copy_to_real(&p);
      FRONT_LH(rc = REAL(cudaGraphExecMemcpyNodeSetParams)(real, node, &p));
      break;
    }
    case cudaGraphNodeTypeMemset:
      FRONT_LH(rc = REAL(cudaGraphExecMemsetNodeSetParams)(real, node,
                                                           &s.set));
      break;
    case cudaGraphNodeTypeHost: {
      cudaHostNodeParams p = { (cudaHostFn_t)s.host_fn, s.host_data };
      FRONT_LH(rc = REAL(cudaGraphExecHostNodeSetParams)(real, node, &p));
      break;
    }
    case cudaGraphNodeTypeEventRecord:
      FRONT_LH(rc = REAL(cudaGraphExecEventRecordNodeSetEvent)(
                 real, node, (cudaEvent_t)s.event->real));
      break;
    case cudaGraphNodeTypeWaitEvent:
      FRONT_LH(rc = REAL(cudaGraphExecEventWaitNodeSetEvent)(
                 real, node, (cudaEvent_t)s.event->real));
      break;
    case cudaGraphNodeTypeGraph:
      FRONT_LH(rc = REAL(cudaGraphExecChildGraphNodeSetParams)(
                 real, node, (cudaGraph_t)s.child_clone->real));
      break;
    default:
      break;
  }
  return rc;
}

// A node update of an executable graph (cudaGraphExec*NodeSetParams): to
// the shadow's node, and kept.
cudaError_t
exec_set(cudaGraphExec_t exec, cudaGraphNode_t node, NodeSnap s)
{
  Exec *e = exec_of(exec);
  GNode *n = e != NULL && node != NULL ? shadow_node(e, node) : NULL;
  if (n == NULL) {
    destroy_clone(s.child_clone);
    return cudaErrorInvalidValue;
  }
  s.node = n;
  cudaError_t rc = apply_override((cudaGraphExec_t)e->real,
                                  (cudaGraphNode_t)n->real, s);
  if (rc != cudaSuccess) {
    destroy_clone(s.child_clone);
    return rc;
  }
  auto it = e->overrides.find(n);
  if (it != e->overrides.end()) {
    destroy_clone(it->second.child_clone);
    e->overrides.erase(it);
  }
  if (s.type == cudaGraphNodeTypeEventRecord && s.event != NULL) {
    e->records.push_back(s.event);
  }
  e->overrides.emplace(n, std::move(s));
  return rc;
}

// A kernel launch's parameters, kept: the function must be one that the
// application registered.
bool
kernel_override(const cudaKernelNodeParams *p, NodeSnap *s)
{
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    if (p == NULL || !front_registered_kernel(p->func)) {
      return false;
    }
  }
  s->type = cudaGraphNodeTypeKernel;
  s->driver = false;
  s->func = p->func;
  s->grid = p->gridDim;
  s->block = p->blockDim;
  s->shared = p->sharedMemBytes;
  cudaFunction_t f = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGetFuncBySymbol)(&f, p->func));
  if (rc != cudaSuccess) {
    return false;
  }
  copy_args((CUfunction)f, p->kernelParams, p->extra, s);
  return true;
}

// A 1D copy, as the 3D copy that a graph's copy node reads back as.
cudaMemcpy3DParms
copy_1d(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
  cudaMemcpy3DParms p = cudaMemcpy3DParms();
  p.srcPtr.ptr = (void *)src;
  p.srcPtr.pitch = p.srcPtr.xsize = count;
  p.srcPtr.ysize = 1;
  p.dstPtr.ptr = dst;
  p.dstPtr.pitch = p.dstPtr.xsize = count;
  p.dstPtr.ysize = 1;
  p.extent.width = count;
  p.extent.height = p.extent.depth = 1;
  p.kind = kind;
  return p;
}

void
forget_exec(Exec *e)
{
  drop_overrides(e);
  destroy_clone(e->shadow);
  destroy_clone(e->update);
  std::lock_guard<std::mutex> guard(objs_lock);
  obj_unlink(e);
  delete e;
}

// Instantiates 'graph' through a clone; 'call' instantiates it.
template <typename Call>
cudaError_t
instantiate(cudaGraphExec_t *out, cudaGraph_t graph,
            unsigned long long flags, Call call)
{
  if (out == NULL) {
    return cudaErrorInvalidValue;
  }
  if (flags & cudaGraphInstantiateFlagDeviceLaunch) {
    front_fatal("graphs for device launch are not supported\n");
  }
  if (graph == NULL) {
    return cudaErrorInvalidValue;
  }
  Graph *t = graph_of(graph);
  Graph *shadow = NULL;
  cudaError_t rc = clone_for_front((cudaGraph_t)t->real, &shadow);
  if (rc != cudaSuccess) {
    return rc;
  }
  cudaGraphExec_t real = NULL;
  rc = call(&real, (cudaGraph_t)shadow->real);
  if (rc != cudaSuccess) {
    destroy_clone(shadow);
    return rc;
  }
  // The template's nodes and the shadow's: the same order (a probe).
  size_t n = 0;
  FRONT_LH(rc = REAL(cudaGraphGetNodes)((cudaGraph_t)t->real, NULL, &n));
  std::vector<cudaGraphNode_t> tn(n), sn(n);
  size_t m = n;
  if (n > 0) {
    FRONT_LH(rc = REAL(cudaGraphGetNodes)((cudaGraph_t)t->real, tn.data(),
                                          &n));
    FRONT_LH(rc = REAL(cudaGraphGetNodes)((cudaGraph_t)shadow->real,
                                          sn.data(), &m));
  }
  Exec *e = new Exec();
  e->real = real;
  e->flags = flags;
  e->shadow = shadow;
  e->ctx = front_object_ctx();
  collect_records((cudaGraph_t)shadow->real, &e->records);
  std::lock_guard<std::mutex> guard(objs_lock);
  for (size_t i = 0; i < n && i < m; i++) {
    e->of_template[node_rec(tn[i], t)] = node_rec(sn[i], shadow);
  }
  obj_link(e, FRONT_GEXEC);
  *out = (cudaGraphExec_t)e;
  return cudaSuccess;
}

// A launch records its event record nodes' events: their times are real
// again (front_async.cpp).
void
note_launch(Exec *e)
{
  for (FrontObj *o : e->records) {
    if (o->magic != FRONT_MAGIC) {
      continue;  // destroyed since
    }
    ((Event *)o)->time_saved = false;
    ((Event *)o)->ckpt_state = 0;
  }
}

// ---- Stream capture ------------------------------------------------------

template <typename Begin>
cudaError_t
begin_capture(cudaStream_t stream, Begin call)
{
  FRONT_ENTER();
  cudaStream_t real = front_real_stream(stream);
  cudaError_t rc = call(real);
  if (rc != cudaSuccess) {
    FRONT_LEAVE();
    return rc;
  }
  cudaStreamCaptureStatus status;
  unsigned long long id = 0;
  cudaError_t got;
  FRONT_LH(got = REAL(cudaStreamGetCaptureInfo)(real, &status, &id, NULL,
                                                NULL, NULL, NULL));
  (void)got;
  std::lock_guard<std::mutex> guard(objs_lock);
  capturing[stream] = Capture{ &front_thread, id };
  __atomic_add_fetch(&front_captures, 1, __ATOMIC_ACQ_REL);
  return rc;  // the gate stays held: FRONT_LEAVE() in end_capture()
}

template <typename End>
cudaError_t
end_capture(cudaStream_t stream, cudaGraph_t *graph, End call)
{
  FRONT_ENTER();
  cudaStream_t real = front_real_stream(stream);
  cudaGraph_t g = NULL;
  cudaError_t rc = call(real, graph != NULL ? &g : NULL);
  bool ended = !front_capturing(real);
  bool held = false;
  Ctx *ctx = rc == cudaSuccess && g != NULL ? front_object_ctx() : NULL;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = capturing.find(stream);
    if (ended && it != capturing.end()) {
      if (it->second.thread != &front_thread) {
        front_fatal("a stream capture ended on another thread than the one "
                    "that began it\n");
      }
      auto made = capture_graphs.find(it->second.id);
      if (made != capture_graphs.end()) {
        if (rc != cudaSuccess || g != made->second->real) {
          forget_graph(made->second);  // the driver's is gone
        }
        capture_graphs.erase(made);
      }
      capturing.erase(it);
      held = true;
    }
    if (rc == cudaSuccess && graph != NULL) {
      *graph = (cudaGraph_t)graph_rec(g, APP, ctx);
    }
  }
  if (held) {
    __atomic_sub_fetch(&front_captures, 1, __ATOMIC_ACQ_REL);
    FRONT_LEAVE();  // begin_capture()'s
  }
  FRONT_LEAVE();
  return rc;
}

template <typename Info>
cudaError_t
capture_info(cudaStream_t stream, cudaStreamCaptureStatus *status,
             unsigned long long *id, cudaGraph_t *graph,
             const cudaGraphNode_t **deps, const cudaGraphEdgeData **edges,
             size_t *n, Info call)
{
  static __thread std::vector<cudaGraphNode_t> *mine;
  FRONT_ENTER();
  cudaGraph_t g = NULL;
  const cudaGraphNode_t *d = NULL;
  size_t count = 0;
  unsigned long long capture_id = 0;
  cudaError_t rc = call(front_real_stream(stream), status, &capture_id, &g,
                        deps != NULL ? &d : NULL, edges,
                        n != NULL ? &count : NULL);
  if (rc == cudaSuccess) {
    if (id != NULL) {
      *id = capture_id;
    }
    Ctx *ctx = g != NULL ? front_object_ctx() : NULL;
    std::lock_guard<std::mutex> guard(objs_lock);
    bool known = g == NULL || graph_by_real.count(g) > 0;
    Graph *rec = graph_rec(g, APP, ctx);
    if (!known) {
      capture_graphs[capture_id] = rec;
    }
    if (graph != NULL) {
      *graph = (cudaGraph_t)rec;
    }
    if (deps != NULL) {
      if (mine == NULL) {
        mine = new std::vector<cudaGraphNode_t>();
      }
      mine->clear();
      for (size_t i = 0; d != NULL && i < count; i++) {
        mine->push_back((cudaGraphNode_t)node_rec(d[i], rec));
      }
      *deps = mine->empty() ? NULL : mine->data();
    }
    if (n != NULL) {
      *n = count;
    }
  }
  FRONT_LEAVE();
  return rc;
}

// ---- At the checkpoint and the restart -----------------------------------

void
graphs_stash()
{
  if (__atomic_load_n(&front_captures, __ATOMIC_ACQUIRE) > 0) {
    front_fatal("a stream capture is in progress at the checkpoint\n");
  }
  std::vector<Graph *> graphs;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    for (FrontObj *o = obj_heads[FRONT_GRAPH]; o != NULL; o = o->next) {
      if (((Graph *)o)->owner != EMBEDDED) {
        graphs.push_back((Graph *)o);
      }
    }
  }
  for (Graph *g : graphs) {
    delete g->snap;
    g->snap = new GraphSnap();
    snap_graph(g, g->snap);
  }
}

void
graphs_drop()
{
  std::lock_guard<std::mutex> guard(objs_lock);
  for (FrontObj *o = obj_heads[FRONT_GRAPH]; o != NULL; o = o->next) {
    delete ((Graph *)o)->snap;
    ((Graph *)o)->snap = NULL;
  }
}

// After the replay: the graphs in creation order, then the executable
// graphs.
void
graphs_settle()
{
  for (FrontObj *o = obj_heads[FRONT_GRAPH]; o != NULL; o = o->next) {
    Graph *g = (Graph *)o;
    if (g->owner == EMBEDDED) {
      continue;
    }
    if (g->snap == NULL) {
      front_fatal("a graph without a snapshot at the restart\n");
    }
    if (g->ctx != NULL) {
      front_ctx_use(g->ctx);
    }
    cudaGraph_t real = NULL;
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaGraphCreate)(&real, 0));
    if (rc != cudaSuccess) {
      front_fatal("making a graph again: error %d\n", rc);
    }
    g->real = real;
    build(*g->snap, real, g->ctx);
  }
  for (FrontObj *o = obj_heads[FRONT_GEXEC]; o != NULL; o = o->next) {
    Exec *e = (Exec *)o;
    e->real = NULL;
    if (e->gone) {
      continue;
    }
    front_ctx_use(e->ctx);
    cudaGraphExec_t real = NULL;
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaGraphInstantiateWithFlags)(
               &real, (cudaGraph_t)e->shadow->real, e->flags));
    if (rc == cudaSuccess && e->update != NULL) {
      cudaGraphExecUpdateResultInfo info;
      FRONT_LH(rc = REAL(cudaGraphExecUpdate)(
                 real, (cudaGraph_t)e->update->real, &info));
    }
    for (auto &entry : e->overrides) {
      if (rc == cudaSuccess) {
        rc = apply_override(real, (cudaGraphNode_t)entry.first->real,
                            entry.second);
      }
    }
    for (auto &entry : e->enabled) {
      if (rc == cudaSuccess) {
        FRONT_LH(rc = REAL(cudaGraphNodeSetEnabled)(
                   real, (cudaGraphNode_t)entry.first->real, entry.second));
      }
    }
    if (rc != cudaSuccess) {
      front_fatal("instantiating a graph again: error %d\n", rc);
    }
    e->real = real;
  }
  graph_by_real.clear();
  node_by_real.clear();
  for (FrontObj *o = obj_heads[FRONT_GRAPH]; o != NULL; o = o->next) {
    graph_by_real[o->real] = (Graph *)o;
  }
  for (FrontObj *o = obj_heads[FRONT_GNODE]; o != NULL; o = o->next) {
    node_by_real[o->real] = (GNode *)o;
  }
}

void
graphs_restore(FrontObj *)
{
}

void
graphs_forget(Ctx *ctx)
{
  for (FrontObj *o = obj_heads[FRONT_GEXEC]; o != NULL; o = o->next) {
    if (((Exec *)o)->ctx == ctx) {
      ((Exec *)o)->gone = true;
    }
  }
}

const FrontLibRecords graph_records = { graphs_restore, graphs_settle,
                                        graphs_forget, graphs_stash,
                                        graphs_drop };

__attribute__((constructor)) void
register_records()
{
  front_lib_records(FRONT_GRAPH, &graph_records);
}

// A node added to 'graph' after 'deps': 'call' adds it with real handles.
template <typename Add>
cudaError_t
add_node(cudaGraphNode_t *node, cudaGraph_t graph,
         const cudaGraphNode_t *deps, size_t n, Add call)
{
  if (node == NULL || graph == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  Graph *g = graph_of(graph);
  std::vector<cudaGraphNode_t> real_deps = real_nodes(deps, n);
  cudaGraphNode_t real = NULL;
  cudaError_t rc = call(&real, (cudaGraph_t)g->real,
                        deps != NULL ? real_deps.data() : NULL, n);
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    *node = (cudaGraphNode_t)node_rec(real, g);
  }
  FRONT_LEAVE();
  return rc;
}

// Nodes listed by 'call' into 'nodes' (of capacity *n), as records of 'g'.
template <typename List>
cudaError_t
list_nodes(cudaGraphNode_t *nodes, size_t *n, Graph *g, List call)
{
  FRONT_ENTER();
  size_t capacity = n != NULL ? *n : 0;
  cudaError_t rc = call();
  if (rc == cudaSuccess && n != NULL) {
    virtual_nodes(nodes, std::min(capacity, *n), g);
  }
  FRONT_LEAVE();
  return rc;
}

}  // namespace

bool
front_capturing(cudaStream_t real)
{
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaStreamIsCapturing)(real, &status));
  return rc == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

// ---- Stream capture ------------------------------------------------------

extern "C" cudaError_t
cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)
{
  return begin_capture(stream, [&](cudaStream_t real) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaStreamBeginCapture)(real, mode));
    return rc;
  });
}

extern "C" cudaError_t
cudaStreamBeginCapture_ptsz(cudaStream_t stream,
                            enum cudaStreamCaptureMode mode)
{
  return begin_capture(stream, [&](cudaStream_t real) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaStreamBeginCapture_ptsz)(real, mode));
    return rc;
  });
}

extern "C" cudaError_t
cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph,
                              const cudaGraphNode_t *dependencies,
                              const cudaGraphEdgeData *dependencyData,
                              size_t numDependencies,
                              enum cudaStreamCaptureMode mode)
{
  cudaGraph_t g = real_graph(graph);
  std::vector<cudaGraphNode_t> deps = real_nodes(dependencies,
                                                 numDependencies);
  return begin_capture(stream, [&](cudaStream_t real) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaStreamBeginCaptureToGraph)(
               real, g, dependencies != NULL ? deps.data() : NULL,
               dependencyData, numDependencies, mode));
    return rc;
  });
}

extern "C" cudaError_t
cudaStreamBeginCaptureToGraph_ptsz(cudaStream_t stream, cudaGraph_t graph,
                                   const cudaGraphNode_t *dependencies,
                                   const cudaGraphEdgeData *dependencyData,
                                   size_t numDependencies,
                                   enum cudaStreamCaptureMode mode)
{
  cudaGraph_t g = real_graph(graph);
  std::vector<cudaGraphNode_t> deps = real_nodes(dependencies,
                                                 numDependencies);
  return begin_capture(stream, [&](cudaStream_t real) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaStreamBeginCaptureToGraph_ptsz)(
               real, g, dependencies != NULL ? deps.data() : NULL,
               dependencyData, numDependencies, mode));
    return rc;
  });
}

extern "C" cudaError_t
cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
  return end_capture(stream, pGraph, [](cudaStream_t real, cudaGraph_t *g) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaStreamEndCapture)(real, g));
    return rc;
  });
}

extern "C" cudaError_t
cudaStreamEndCapture_ptsz(cudaStream_t stream, cudaGraph_t *pGraph)
{
  return end_capture(stream, pGraph, [](cudaStream_t real, cudaGraph_t *g) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaStreamEndCapture_ptsz)(real, g));
    return rc;
  });
}

extern "C" cudaError_t
cudaStreamGetCaptureInfo(cudaStream_t stream,
                         enum cudaStreamCaptureStatus *captureStatus_out,
                         unsigned long long *id_out, cudaGraph_t *graph_out,
                         const cudaGraphNode_t **dependencies_out,
                         const cudaGraphEdgeData **edgeData_out,
                         size_t *numDependencies_out)
{
  return capture_info(
    stream, captureStatus_out, id_out, graph_out, dependencies_out,
    edgeData_out, numDependencies_out,
    [](cudaStream_t real, cudaStreamCaptureStatus *s, unsigned long long *id,
       cudaGraph_t *g, const cudaGraphNode_t **d,
       const cudaGraphEdgeData **e, size_t *n) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaStreamGetCaptureInfo)(real, s, id, g, d, e, n));
      return rc;
    });
}

extern "C" cudaError_t
cudaStreamGetCaptureInfo_ptsz(cudaStream_t stream,
                              enum cudaStreamCaptureStatus *captureStatus_out,
                              unsigned long long *id_out,
                              cudaGraph_t *graph_out,
                              const cudaGraphNode_t **dependencies_out,
                              const cudaGraphEdgeData **edgeData_out,
                              size_t *numDependencies_out)
{
  return capture_info(
    stream, captureStatus_out, id_out, graph_out, dependencies_out,
    edgeData_out, numDependencies_out,
    [](cudaStream_t real, cudaStreamCaptureStatus *s, unsigned long long *id,
       cudaGraph_t *g, const cudaGraphNode_t **d,
       const cudaGraphEdgeData **e, size_t *n) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaStreamGetCaptureInfo_ptsz)(real, s, id, g, d, e,
                                                        n));
      return rc;
    });
}

extern "C" cudaError_t
cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                    cudaGraphNode_t *dependencies,
                                    const cudaGraphEdgeData *dependencyData,
                                    size_t numDependencies, unsigned int flags)
{
  FRONT_ENTER();
  std::vector<cudaGraphNode_t> deps = real_nodes(dependencies,
                                                 numDependencies);
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaStreamUpdateCaptureDependencies)(
             front_real_stream(stream),
             dependencies != NULL ? deps.data() : NULL, dependencyData,
             numDependencies, flags));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaStreamUpdateCaptureDependencies_ptsz(
  cudaStream_t stream, cudaGraphNode_t *dependencies,
  const cudaGraphEdgeData *dependencyData, size_t numDependencies,
  unsigned int flags)
{
  FRONT_ENTER();
  std::vector<cudaGraphNode_t> deps = real_nodes(dependencies,
                                                 numDependencies);
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaStreamUpdateCaptureDependencies_ptsz)(
             front_real_stream(stream),
             dependencies != NULL ? deps.data() : NULL, dependencyData,
             numDependencies, flags));
  FRONT_LEAVE();
  return rc;
}

// ---- Graphs --------------------------------------------------------------

extern "C" cudaError_t
cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags)
{
  if (pGraph == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  cudaGraph_t real = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphCreate)(&real, flags));
  if (rc == cudaSuccess) {
    Ctx *ctx = front_object_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    *pGraph = (cudaGraph_t)graph_rec(real, APP, ctx);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph)
{
  if (pGraphClone == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  cudaGraph_t real = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphClone)(&real, real_graph(originalGraph)));
  if (rc == cudaSuccess) {
    Ctx *ctx = front_object_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    *pGraphClone = (cudaGraph_t)graph_rec(real, APP, ctx);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphDestroy(cudaGraph_t graph)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphDestroy)(real_graph(graph)));
  if (rc == cudaSuccess && graph != NULL) {
    Graph *g = graph_of(graph);
    std::lock_guard<std::mutex> guard(objs_lock);
    forget_graph(g);
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Nodes ---------------------------------------------------------------

extern "C" cudaError_t
cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaKernelNodeParams *pNodeParams)
{
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddKernelNode)(n, g, d, nd,
                                                               pNodeParams));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaMemcpy3DParms *pCopyParams)
{
  if (pCopyParams == NULL) {
    return cudaErrorInvalidValue;
  }
  cudaMemcpy3DParms p = *pCopyParams;
  copy_to_real(&p);
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddMemcpyNode)(n, g, d, nd,
                                                               &p));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                         const cudaGraphNode_t *pDependencies,
                         size_t numDependencies, void *dst, const void *src,
                         size_t count, enum cudaMemcpyKind kind)
{
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddMemcpyNode1D)(
                               n, g, d, nd, dst, src, count, kind));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *pGraphNode,
                               cudaGraph_t graph,
                               const cudaGraphNode_t *pDependencies,
                               size_t numDependencies, const void *symbol,
                               const void *src, size_t count, size_t offset,
                               enum cudaMemcpyKind kind)
{
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddMemcpyNodeToSymbol)(
                               n, g, d, nd, symbol, src, count, offset,
                               kind));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *pGraphNode,
                                 cudaGraph_t graph,
                                 const cudaGraphNode_t *pDependencies,
                                 size_t numDependencies, void *dst,
                                 const void *symbol, size_t count,
                                 size_t offset, enum cudaMemcpyKind kind)
{
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddMemcpyNodeFromSymbol)(
                               n, g, d, nd, dst, symbol, count, offset,
                               kind));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaMemsetParams *pMemsetParams)
{
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddMemsetNode)(
                               n, g, d, nd, pMemsetParams));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                     const cudaGraphNode_t *pDependencies,
                     size_t numDependencies,
                     const struct cudaHostNodeParams *pNodeParams)
{
  if (pNodeParams == NULL || pNodeParams->fn == NULL) {
    return cudaErrorInvalidValue;
  }
  cudaHostNodeParams p;
  void *fn;
  front_host_node(pNodeParams->fn, pNodeParams->userData, &fn, &p.userData);
  p.fn = (cudaHostFn_t)fn;
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddHostNode)(n, g, d, nd,
                                                             &p));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                      const cudaGraphNode_t *pDependencies,
                      size_t numDependencies)
{
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddEmptyNode)(n, g, d, nd));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                           const cudaGraphNode_t *pDependencies,
                           size_t numDependencies, cudaGraph_t childGraph)
{
  cudaGraph_t child = real_graph(childGraph);
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddChildGraphNode)(
                               n, g, d, nd, child));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                            const cudaGraphNode_t *pDependencies,
                            size_t numDependencies, cudaEvent_t event)
{
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddEventRecordNode)(
                               n, g, d, nd, front_real_event(event)));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                          const cudaGraphNode_t *pDependencies,
                          size_t numDependencies, cudaEvent_t event)
{
  return add_node(pGraphNode, graph, pDependencies, numDependencies,
                  [&](cudaGraphNode_t *n, cudaGraph_t g,
                      const cudaGraphNode_t *d, size_t nd) {
                    cudaError_t rc;
                    FRONT_LH(rc = REAL(cudaGraphAddEventWaitNode)(
                               n, g, d, nd, front_real_event(event)));
                    return rc;
                  });
}

extern "C" cudaError_t
cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes,
                  size_t *numNodes)
{
  if (graph == NULL) {
    return cudaErrorInvalidValue;
  }
  Graph *g = graph_of(graph);
  return list_nodes(nodes, numNodes, g, [&]() {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaGraphGetNodes)((cudaGraph_t)g->real, nodes,
                                          numNodes));
    return rc;
  });
}

extern "C" cudaError_t
cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes,
                      size_t *pNumRootNodes)
{
  if (graph == NULL) {
    return cudaErrorInvalidValue;
  }
  Graph *g = graph_of(graph);
  return list_nodes(pRootNodes, pNumRootNodes, g, [&]() {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaGraphGetRootNodes)((cudaGraph_t)g->real,
                                              pRootNodes, pNumRootNodes));
    return rc;
  });
}

extern "C" cudaError_t
cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from,
                  cudaGraphNode_t *to, cudaGraphEdgeData *edgeData,
                  size_t *numEdges)
{
  if (graph == NULL) {
    return cudaErrorInvalidValue;
  }
  Graph *g = graph_of(graph);
  FRONT_ENTER();
  size_t capacity = numEdges != NULL ? *numEdges : 0;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphGetEdges)((cudaGraph_t)g->real, from, to,
                                        edgeData, numEdges));
  if (rc == cudaSuccess && numEdges != NULL) {
    virtual_nodes(from, std::min(capacity, *numEdges), g);
    virtual_nodes(to, std::min(capacity, *numEdges), g);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphNodeGetDependencies(cudaGraphNode_t node,
                             cudaGraphNode_t *pDependencies,
                             cudaGraphEdgeData *edgeData,
                             size_t *pNumDependencies)
{
  if (node == NULL) {
    return cudaErrorInvalidValue;
  }
  GNode *n = node_of(node);
  return list_nodes(pDependencies, pNumDependencies, n->graph, [&]() {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaGraphNodeGetDependencies)(
               (cudaGraphNode_t)n->real, pDependencies, edgeData,
               pNumDependencies));
    return rc;
  });
}

extern "C" cudaError_t
cudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
                               cudaGraphNode_t *pDependentNodes,
                               cudaGraphEdgeData *edgeData,
                               size_t *pNumDependentNodes)
{
  if (node == NULL) {
    return cudaErrorInvalidValue;
  }
  GNode *n = node_of(node);
  return list_nodes(pDependentNodes, pNumDependentNodes, n->graph, [&]() {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaGraphNodeGetDependentNodes)(
               (cudaGraphNode_t)n->real, pDependentNodes, edgeData,
               pNumDependentNodes));
    return rc;
  });
}

extern "C" cudaError_t
cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode,
                         cudaGraph_t clonedGraph)
{
  if (pNode == NULL || clonedGraph == NULL) {
    return cudaErrorInvalidValue;
  }
  Graph *g = graph_of(clonedGraph);
  FRONT_ENTER();
  cudaGraphNode_t real = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphNodeFindInClone)(
             &real, real_node(originalNode), (cudaGraph_t)g->real));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    *pNode = (cudaGraphNode_t)node_rec(real, g);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode, cudaGraph_t *phGraph)
{
  if (phGraph == NULL || hNode == NULL) {
    return cudaErrorInvalidValue;
  }
  *phGraph = (cudaGraph_t)node_of(hNode)->graph;
  return cudaSuccess;
}

extern "C" cudaError_t
cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph)
{
  if (pGraph == NULL || node == NULL) {
    return cudaErrorInvalidValue;
  }
  GNode *n = node_of(node);
  FRONT_ENTER();
  cudaGraph_t real = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphChildGraphNodeGetGraph)(
             (cudaGraphNode_t)n->real, &real));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    n->child = graph_rec(real, EMBEDDED, n->graph->ctx);
    *pGraph = (cudaGraph_t)n->child;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphDestroyNode(cudaGraphNode_t node)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphDestroyNode)(real_node(node)));
  if (rc == cudaSuccess && node != NULL) {
    GNode *n = node_of(node);
    std::lock_guard<std::mutex> guard(objs_lock);
    forget_node(n, true);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from,
                         const cudaGraphNode_t *to,
                         const cudaGraphEdgeData *edgeData,
                         size_t numDependencies)
{
  FRONT_ENTER();
  std::vector<cudaGraphNode_t> f = real_nodes(from, numDependencies),
                               t = real_nodes(to, numDependencies);
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphAddDependencies)(
             real_graph(graph), from != NULL ? f.data() : NULL,
             to != NULL ? t.data() : NULL, edgeData, numDependencies));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from,
                            const cudaGraphNode_t *to,
                            const cudaGraphEdgeData *edgeData,
                            size_t numDependencies)
{
  FRONT_ENTER();
  std::vector<cudaGraphNode_t> f = real_nodes(from, numDependencies),
                               t = real_nodes(to, numDependencies);
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphRemoveDependencies)(
             real_graph(graph), from != NULL ? f.data() : NULL,
             to != NULL ? t.data() : NULL, edgeData, numDependencies));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,
                             struct cudaMemcpy3DParms *pNodeParams)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphMemcpyNodeGetParams)(real_node(node),
                                                   pNodeParams));
  if (rc == cudaSuccess && pNodeParams != NULL) {
    copy_to_records(pNodeParams);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node,
                             const struct cudaMemcpy3DParms *pNodeParams)
{
  if (pNodeParams == NULL) {
    return cudaErrorInvalidValue;
  }
  cudaMemcpy3DParms p = *pNodeParams;
  copy_to_real(&p);
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphMemcpyNodeSetParams)(real_node(node), &p));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphHostNodeGetParams(cudaGraphNode_t node,
                           struct cudaHostNodeParams *pNodeParams)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphHostNodeGetParams)(real_node(node),
                                                 pNodeParams));
  if (rc == cudaSuccess && pNodeParams != NULL) {
    front_host_node_target((void *)pNodeParams->fn, pNodeParams->userData,
                           &pNodeParams->fn, &pNodeParams->userData);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphHostNodeSetParams(cudaGraphNode_t node,
                           const struct cudaHostNodeParams *pNodeParams)
{
  if (pNodeParams == NULL || pNodeParams->fn == NULL) {
    return cudaErrorInvalidValue;
  }
  cudaHostNodeParams p;
  void *fn;
  front_host_node(pNodeParams->fn, pNodeParams->userData, &fn, &p.userData);
  p.fn = (cudaHostFn_t)fn;
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphHostNodeSetParams)(real_node(node), &p));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphEventRecordNodeGetEvent)(real_node(node),
                                                       event_out));
  if (rc == cudaSuccess && event_out != NULL) {
    *event_out = (cudaEvent_t)event_record(*event_out);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphEventWaitNodeGetEvent)(real_node(node),
                                                     event_out));
  if (rc == cudaSuccess && event_out != NULL) {
    *event_out = (cudaEvent_t)event_record(*event_out);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode,
                                cudaLaunchAttributeID attr,
                                const cudaLaunchAttributeValue *value)
{
  if (attr == cudaLaunchAttributeProgrammaticEvent ||
      attr == cudaLaunchAttributeLaunchCompletionEvent ||
      (attr == cudaLaunchAttributeDeviceUpdatableKernelNode && value != NULL &&
       value->deviceUpdatableKernelNode.deviceUpdatable)) {
    front_fatal("kernel node attribute %d (events, device-updatable nodes) "
                "is not supported\n", (int)attr);
  }
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphKernelNodeSetAttribute)(real_node(hNode), attr,
                                                      value));
  FRONT_LEAVE();
  return rc;
}

// ---- Executable graphs ---------------------------------------------------

extern "C" cudaError_t
cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                     unsigned long long flags)
{
  FRONT_ENTER();
  cudaError_t rc = instantiate(
    pGraphExec, graph, flags, [&](cudaGraphExec_t *x, cudaGraph_t g) {
      cudaError_t r;
      FRONT_LH(r = REAL(cudaGraphInstantiate)(x, g, flags));
      return r;
    });
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                              unsigned long long flags)
{
  FRONT_ENTER();
  cudaError_t rc = instantiate(
    pGraphExec, graph, flags, [&](cudaGraphExec_t *x, cudaGraph_t g) {
      cudaError_t r;
      FRONT_LH(r = REAL(cudaGraphInstantiateWithFlags)(x, g, flags));
      return r;
    });
  FRONT_LEAVE();
  return rc;
}

// The error node, if any, is the clone's: the template's at its index.
template <typename Instantiate>
cudaError_t
instantiate_with_params(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                        cudaGraphInstantiateParams *params,
                        Instantiate call)
{
  if (params == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  cudaGraphInstantiateParams p = *params;
  p.uploadStream = front_real_stream(params->uploadStream);
  cudaError_t rc = instantiate(
    pGraphExec, graph, p.flags, [&](cudaGraphExec_t *x, cudaGraph_t g) {
      cudaError_t r = call(x, g, &p);
      params->result_out = p.result_out;
      params->errNode_out = NULL;
      if (r != cudaSuccess && p.errNode_out != NULL) {
        size_t n = 0;
        std::vector<cudaGraphNode_t> sn, tn;
        cudaError_t e;
        FRONT_LH(e = REAL(cudaGraphGetNodes)(g, NULL, &n));
        sn.resize(n);
        tn.resize(n);
        FRONT_LH(e = REAL(cudaGraphGetNodes)(g, sn.data(), &n));
        FRONT_LH(e = REAL(cudaGraphGetNodes)(real_graph(graph), tn.data(),
                                             &n));
        (void)e;
        for (size_t i = 0; i < n; i++) {
          if (sn[i] == p.errNode_out) {
            std::lock_guard<std::mutex> guard(objs_lock);
            params->errNode_out =
              (cudaGraphNode_t)node_rec(tn[i], graph_of(graph));
          }
        }
      }
      return r;
    });
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphInstantiateWithParams(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                               cudaGraphInstantiateParams *instantiateParams)
{
  return instantiate_with_params(
    pGraphExec, graph, instantiateParams,
    [](cudaGraphExec_t *x, cudaGraph_t g, cudaGraphInstantiateParams *p) {
      cudaError_t r;
      FRONT_LH(r = REAL(cudaGraphInstantiateWithParams)(x, g, p));
      return r;
    });
}

extern "C" cudaError_t
cudaGraphInstantiateWithParams_ptsz(
  cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
  cudaGraphInstantiateParams *instantiateParams)
{
  return instantiate_with_params(
    pGraphExec, graph, instantiateParams,
    [](cudaGraphExec_t *x, cudaGraph_t g, cudaGraphInstantiateParams *p) {
      cudaError_t r;
      FRONT_LH(r = REAL(cudaGraphInstantiateWithParams_ptsz)(x, g, p));
      return r;
    });
}

extern "C" cudaError_t
cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphExecDestroy)(
             (cudaGraphExec_t)front_real_obj(graphExec, FRONT_GEXEC)));
  if (rc == cudaSuccess && graphExec != NULL) {
    forget_exec(exec_of(graphExec));
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphLaunch)(
             (cudaGraphExec_t)front_real_obj(graphExec, FRONT_GEXEC),
             front_real_stream(stream)));
  if (rc == cudaSuccess && graphExec != NULL) {
    note_launch(exec_of(graphExec));
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphLaunch_ptsz(cudaGraphExec_t graphExec, cudaStream_t stream)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphLaunch_ptsz)(
             (cudaGraphExec_t)front_real_obj(graphExec, FRONT_GEXEC),
             front_real_stream(stream)));
  if (rc == cudaSuccess && graphExec != NULL) {
    note_launch(exec_of(graphExec));
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                    cudaGraphExecUpdateResultInfo *resultInfo)
{
  if (hGraphExec == NULL || hGraph == NULL) {
    return cudaErrorInvalidValue;
  }
  Exec *e = exec_of(hGraphExec);
  Graph *g = graph_of(hGraph);
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGraphExecUpdate)((cudaGraphExec_t)e->real,
                                          (cudaGraph_t)g->real, resultInfo));
  if (resultInfo != NULL) {
    cudaGraphNode_t nodes[2] = { resultInfo->errorNode,
                                 resultInfo->errorFromNode };
    virtual_nodes(nodes, 2, g);
    resultInfo->errorNode = nodes[0];
    resultInfo->errorFromNode = nodes[1];
  }
  if (rc == cudaSuccess) {
    Graph *update = NULL;
    cudaError_t cloned = clone_for_front((cudaGraph_t)g->real, &update);
    if (cloned != cudaSuccess) {
      front_fatal("cloning a graph for an update: error %d\n", cloned);
    }
    destroy_clone(e->update);
    e->update = update;
    drop_overrides(e);
    e->records.clear();
    collect_records((cudaGraph_t)update->real, &e->records);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec,
                                 cudaGraphNode_t node,
                                 const struct cudaKernelNodeParams *pNodeParams)
{
  FRONT_ENTER();
  NodeSnap s = NodeSnap();
  if (!kernel_override(pNodeParams, &s)) {
    front_fatal("an executable graph's kernel node set to a kernel that the "
                "application did not register\n");
  }
  cudaError_t rc = exec_set(hGraphExec, node, std::move(s));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec,
                                 cudaGraphNode_t node,
                                 const struct cudaMemcpy3DParms *pNodeParams)
{
  if (pNodeParams == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  NodeSnap s = NodeSnap();
  s.type = cudaGraphNodeTypeMemcpy;
  s.copy = *pNodeParams;
  cudaError_t rc = exec_set(hGraphExec, node, std::move(s));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec,
                                   cudaGraphNode_t node, void *dst,
                                   const void *src, size_t count,
                                   enum cudaMemcpyKind kind)
{
  FRONT_ENTER();
  NodeSnap s = NodeSnap();
  s.type = cudaGraphNodeTypeMemcpy;
  s.copy = copy_1d(dst, src, count, kind);
  cudaError_t rc = exec_set(hGraphExec, node, std::move(s));
  FRONT_LEAVE();
  return rc;
}

// To and from symbols: the symbol's address is the module data's, which a
// restart keeps (front_objs.cpp).
template <bool to>
cudaError_t
exec_copy_symbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                 void *ptr, const void *symbol, size_t count, size_t offset,
                 enum cudaMemcpyKind kind)
{
  FRONT_ENTER();
  void *address = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGetSymbolAddress)(&address, symbol));
  if (rc == cudaSuccess) {
    NodeSnap s = NodeSnap();
    s.type = cudaGraphNodeTypeMemcpy;
    char *at = (char *)address + offset;
    s.copy = to ? copy_1d(at, ptr, count, kind) : copy_1d(ptr, at, count,
                                                          kind);
    rc = exec_set(hGraphExec, node, std::move(s));
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec,
                                         cudaGraphNode_t node,
                                         const void *symbol, const void *src,
                                         size_t count, size_t offset,
                                         enum cudaMemcpyKind kind)
{
  return exec_copy_symbol<true>(hGraphExec, node, (void *)src, symbol, count,
                                offset, kind);
}

extern "C" cudaError_t
cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec,
                                           cudaGraphNode_t node, void *dst,
                                           const void *symbol, size_t count,
                                           size_t offset,
                                           enum cudaMemcpyKind kind)
{
  return exec_copy_symbol<false>(hGraphExec, node, dst, symbol, count,
                                 offset, kind);
}

extern "C" cudaError_t
cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec,
                                 cudaGraphNode_t node,
                                 const struct cudaMemsetParams *pNodeParams)
{
  if (pNodeParams == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  NodeSnap s = NodeSnap();
  s.type = cudaGraphNodeTypeMemset;
  s.set = *pNodeParams;
  cudaError_t rc = exec_set(hGraphExec, node, std::move(s));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec,
                               cudaGraphNode_t node,
                               const struct cudaHostNodeParams *pNodeParams)
{
  if (pNodeParams == NULL || pNodeParams->fn == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  NodeSnap s = NodeSnap();
  s.type = cudaGraphNodeTypeHost;
  front_host_node(pNodeParams->fn, pNodeParams->userData, &s.host_fn,
                  &s.host_data);
  cudaError_t rc = exec_set(hGraphExec, node, std::move(s));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec,
                                     cudaGraphNode_t node,
                                     cudaGraph_t childGraph)
{
  FRONT_ENTER();
  NodeSnap s = NodeSnap();
  s.type = cudaGraphNodeTypeGraph;
  cudaError_t rc = clone_for_front(real_graph(childGraph), &s.child_clone);
  if (rc == cudaSuccess) {
    rc = exec_set(hGraphExec, node, std::move(s));
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec,
                                     cudaGraphNode_t hNode, cudaEvent_t event)
{
  front_real_event(event);
  FRONT_ENTER();
  NodeSnap s = NodeSnap();
  s.type = cudaGraphNodeTypeEventRecord;
  s.event = (FrontObj *)event;
  cudaError_t rc = exec_set(hGraphExec, hNode, std::move(s));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec,
                                   cudaGraphNode_t hNode, cudaEvent_t event)
{
  FRONT_ENTER();
  NodeSnap s = NodeSnap();
  s.type = cudaGraphNodeTypeWaitEvent;
  s.event = (FrontObj *)event;
  cudaError_t rc = exec_set(hGraphExec, hNode, std::move(s));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode,
                        unsigned int isEnabled)
{
  if (hGraphExec == NULL) {
    return cudaErrorInvalidValue;
  }
  Exec *e = exec_of(hGraphExec);
  FRONT_ENTER();
  GNode *n = shadow_node(e, hNode);
  cudaError_t rc = cudaErrorInvalidValue;
  if (n != NULL) {
    FRONT_LH(rc = REAL(cudaGraphNodeSetEnabled)(
               (cudaGraphExec_t)e->real, (cudaGraphNode_t)n->real, isEnabled));
  }
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    e->enabled[n] = isEnabled;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode,
                        unsigned int *isEnabled)
{
  if (hGraphExec == NULL) {
    return cudaErrorInvalidValue;
  }
  Exec *e = exec_of(hGraphExec);
  FRONT_ENTER();
  GNode *n = shadow_node(e, hNode);
  cudaError_t rc = cudaErrorInvalidValue;
  if (n != NULL) {
    FRONT_LH(rc = REAL(cudaGraphNodeGetEnabled)(
               (cudaGraphExec_t)e->real, (cudaGraphNode_t)n->real, isEnabled));
  }
  FRONT_LEAVE();
  return rc;
}
