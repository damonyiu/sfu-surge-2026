export function dijkstra(graph, startId, endId) {
  if (!graph?.nodes?.length || !graph?.edges?.length) {
    return { path: [], distance: Infinity }
  }

  const distances = new Map()
  const previous = new Map()
  const visited = new Set()
  const adjacency = new Map()

  for (const node of graph.nodes) {
    distances.set(node.id, Infinity)
    adjacency.set(node.id, [])
  }

  for (const edge of graph.edges) {
    if (adjacency.has(edge.from)) {
      adjacency.get(edge.from).push({ to: edge.to, weight: edge.weight })
    }
  }

  if (!distances.has(startId) || !distances.has(endId)) {
    return { path: [], distance: Infinity }
  }

  distances.set(startId, 0)

  while (visited.size < graph.nodes.length) {
    let currentId = null
    let minDistance = Infinity

    for (const [nodeId, distance] of distances) {
      if (!visited.has(nodeId) && distance < minDistance) {
        minDistance = distance
        currentId = nodeId
      }
    }

    if (currentId === null || currentId === endId) {
      break
    }

    visited.add(currentId)

    for (const neighbor of adjacency.get(currentId) ?? []) {
      const candidateDistance = minDistance + neighbor.weight
      if (candidateDistance < distances.get(neighbor.to)) {
        distances.set(neighbor.to, candidateDistance)
        previous.set(neighbor.to, currentId)
      }
    }
  }

  if (distances.get(endId) === Infinity) {
    return { path: [], distance: Infinity }
  }

  const path = []
  let current = endId

  while (current) {
    path.unshift(current)
    current = previous.get(current)
  }

  return { path, distance: distances.get(endId) }
}
