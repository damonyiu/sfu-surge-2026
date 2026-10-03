import { Navigation, Route, RotateCcw, Search, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import graphData from './data/graph.json'
import { dijkstra } from './utils/dijkstra'

type NodeType = {
  id: string
  label: string
  x: number
  y: number
  floor: number
  type: 'classroom' | 'hallway' | 'stairs'
}

type EdgeType = {
  from: string
  to: string
  weight: number
}

type GraphType = {
  nodes: NodeType[]
  edges: EdgeType[]
}

const graph = graphData as GraphType
const defaultStartId = 'main_entrance'

function getTurnByTurn(path: NodeType[]) {
  if (path.length < 2) {
    return []
  }

  const steps: string[] = [`Head to ${path[1].label}`]

  for (let index = 1; index < path.length - 1; index += 1) {
    const previous = path[index - 1]
    const current = path[index]
    const next = path[index + 1]

    const x1 = current.x - previous.x
    const y1 = current.y - previous.y
    const x2 = next.x - current.x
    const y2 = next.y - current.y
    const cross = x1 * y2 - y1 * x2

    if (Math.abs(cross) < 1) {
      steps.push(`Continue through ${current.label}`)
      continue
    }

    steps.push(`Turn ${cross > 0 ? 'left' : 'right'} at ${current.label}`)
  }

  steps.push(`Arrive at ${path[path.length - 1].label}`)
  return steps
}

function App() {
  const nodeById = useMemo(
    () => new Map(graph.nodes.map((node) => [node.id, node])),
    [],
  )

  const allLocations = useMemo(() => [...graph.nodes].sort((a, b) => a.label.localeCompare(b.label)), [])

  const [startId, setStartId] = useState(defaultStartId)
  const [startInput, setStartInput] = useState(nodeById.get(defaultStartId)?.label ?? 'Main Entrance')
  const [destinationId, setDestinationId] = useState('')
  const [destinationInput, setDestinationInput] = useState('')
  const [activeDropdown, setActiveDropdown] = useState<'start' | 'destination' | null>(null)

  const routeResult = useMemo(() => {
    if (!startId || !destinationId || startId === destinationId) {
      return { path: [] as NodeType[], distance: Infinity }
    }

    const result = dijkstra(graph, startId, destinationId)
    return {
      path: result.path.map((id) => nodeById.get(id)).filter(Boolean) as NodeType[],
      distance: result.distance,
    }
  }, [destinationId, nodeById, startId])

  const directions = useMemo(() => getTurnByTurn(routeResult.path), [routeResult.path])

  const routePoints = routeResult.path.map((node) => `${node.x},${node.y}`).join(' ')

  const visibleEdges = useMemo(() => {
    const seen = new Set<string>()
    return graph.edges.filter((edge) => {
      const key = [edge.from, edge.to].sort().join(':')
      if (seen.has(key)) {
        return false
      }

      seen.add(key)
      return true
    })
  }, [])

  const filterLocations = (value: string) => {
    const trimmed = value.trim().toLowerCase()
    if (!trimmed) {
      return allLocations
    }

    return allLocations.filter((node) => node.label.toLowerCase().includes(trimmed))
  }

  const startMatches = filterLocations(startInput).slice(0, 8)
  const destinationMatches = filterLocations(destinationInput).slice(0, 8)

  const selectStart = (node: NodeType) => {
    setStartId(node.id)
    setStartInput(node.label)
    setActiveDropdown(null)
  }

  const selectDestination = (node: NodeType) => {
    setDestinationId(node.id)
    setDestinationInput(node.label)
    setActiveDropdown(null)
  }

  const clearRoute = () => {
    setDestinationId('')
    setDestinationInput('')
    setActiveDropdown(null)
  }

  const resetRoute = () => {
    setStartId(defaultStartId)
    setStartInput(nodeById.get(defaultStartId)?.label ?? 'Main Entrance')
    setDestinationId('')
    setDestinationInput('')
    setActiveDropdown(null)
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <main className="mx-auto flex w-full max-w-7xl flex-col gap-4 p-4 lg:p-6">
        <header className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-4 flex items-center gap-2">
            <Navigation className="h-5 w-5 text-indigo-600" />
            <h1 className="text-xl font-semibold">SFU Indoor Navigator</h1>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-[1fr_1fr_auto]">
            <div className="relative">
              <label className="mb-1 block text-sm font-medium text-slate-600">Start Location</label>
              <div className="flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 focus-within:border-indigo-500">
                <Search className="h-4 w-4 text-slate-400" />
                <input
                  className="w-full border-0 bg-transparent text-sm outline-none"
                  value={startInput}
                  onChange={(event) => {
                    setStartInput(event.target.value)
                    setActiveDropdown('start')
                  }}
                  onFocus={() => setActiveDropdown('start')}
                  onBlur={() => setTimeout(() => setActiveDropdown(null), 120)}
                  placeholder="Main Entrance"
                />
              </div>
              {activeDropdown === 'start' && startMatches.length > 0 ? (
                <ul className="absolute z-10 mt-1 max-h-52 w-full overflow-y-auto rounded-lg border border-slate-200 bg-white p-1 shadow-lg">
                  {startMatches.map((node) => (
                    <li key={node.id}>
                      <button
                        type="button"
                        onClick={() => selectStart(node)}
                        className="w-full rounded-md px-3 py-2 text-left text-sm hover:bg-slate-100"
                      >
                        {node.label}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>

            <div className="relative">
              <label className="mb-1 block text-sm font-medium text-slate-600">Destination</label>
              <div className="flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 focus-within:border-indigo-500">
                <Route className="h-4 w-4 text-slate-400" />
                <input
                  className="w-full border-0 bg-transparent text-sm outline-none"
                  value={destinationInput}
                  onChange={(event) => {
                    setDestinationInput(event.target.value)
                    setDestinationId('')
                    setActiveDropdown('destination')
                  }}
                  onFocus={() => setActiveDropdown('destination')}
                  onBlur={() => setTimeout(() => setActiveDropdown(null), 120)}
                  placeholder="Search classroom or hallway"
                />
              </div>
              {activeDropdown === 'destination' && destinationMatches.length > 0 ? (
                <ul className="absolute z-10 mt-1 max-h-52 w-full overflow-y-auto rounded-lg border border-slate-200 bg-white p-1 shadow-lg">
                  {destinationMatches.map((node) => (
                    <li key={node.id}>
                      <button
                        type="button"
                        onClick={() => selectDestination(node)}
                        className="w-full rounded-md px-3 py-2 text-left text-sm hover:bg-slate-100"
                      >
                        {node.label}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>

            <div className="flex items-end gap-2">
              <button
                type="button"
                onClick={clearRoute}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium hover:bg-slate-100"
              >
                <XCircle className="h-4 w-4" />
                Clear
              </button>
              <button
                type="button"
                onClick={resetRoute}
                className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-500"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
            </div>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[2fr_1fr]">
          <div className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
            <div className="aspect-square w-full overflow-hidden rounded-xl bg-slate-50">
              <svg viewBox="0 0 1000 1000" className="h-full w-full">
                <rect x="0" y="0" width="1000" height="1000" fill="#f8fafc" />
                {visibleEdges.map((edge) => {
                  const fromNode = nodeById.get(edge.from)
                  const toNode = nodeById.get(edge.to)
                  if (!fromNode || !toNode) {
                    return null
                  }

                  return (
                    <line
                      key={`${edge.from}-${edge.to}`}
                      x1={fromNode.x}
                      y1={fromNode.y}
                      x2={toNode.x}
                      y2={toNode.y}
                      stroke="#cbd5e1"
                      strokeWidth="8"
                      strokeLinecap="round"
                    />
                  )
                })}

                {routePoints ? (
                  <polyline
                    points={routePoints}
                    fill="none"
                    stroke="#4f46e5"
                    strokeWidth="14"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="route-dash"
                  />
                ) : null}

                {graph.nodes.map((node) => (
                  <g key={node.id}>
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={node.type === 'classroom' ? 8 : 10}
                      fill={node.type === 'classroom' ? '#0f172a' : node.type === 'stairs' ? '#7c3aed' : '#94a3b8'}
                    />
                    {node.type !== 'hallway' ? (
                      <text x={node.x + 12} y={node.y - 12} fontSize="20" fill="#334155" fontWeight="600">
                        {node.label}
                      </text>
                    ) : null}
                  </g>
                ))}

                {routeResult.path.length > 0 ? (
                  <circle
                    cx={routeResult.path[routeResult.path.length - 1].x}
                    cy={routeResult.path[routeResult.path.length - 1].y}
                    r="16"
                    className="destination-pulse"
                  />
                ) : null}
              </svg>
            </div>
          </div>

          <aside className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-1 text-lg font-semibold">Turn-by-turn Directions</h2>
            {routeResult.path.length > 0 ? (
              <p className="mb-4 text-sm text-slate-500">Estimated distance: {Math.round(routeResult.distance)} units</p>
            ) : (
              <p className="mb-4 text-sm text-slate-500">Select a destination to see route guidance.</p>
            )}

            <ol className="space-y-2 text-sm text-slate-700">
              {directions.map((step, index) => (
                <li key={step} className="rounded-md bg-slate-50 px-3 py-2">
                  <span className="mr-2 inline-flex h-5 w-5 items-center justify-center rounded-full bg-indigo-600 text-xs font-semibold text-white">
                    {index + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          </aside>
        </section>
      </main>
    </div>
  )
}

export default App
