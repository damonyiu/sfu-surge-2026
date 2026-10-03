export function dijkstra(
  graph: {
    nodes: Array<{ id: string }>
    edges: Array<{ from: string; to: string; weight: number }>
  },
  startId: string,
  endId: string,
): { path: string[]; distance: number }
