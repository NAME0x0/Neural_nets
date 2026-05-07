'use client';

import { useMemo } from 'react';
import { ReactFlow, Background, BackgroundVariant, type Edge, type Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useAppStore } from '@/lib/store/use-app-store';

const NODE_X_GAP = 220;
const NODE_Y_GAP = 60;

export function NetworkGraph() {
  const inputSize = useAppStore((s) => s.inputSize);
  const layers = useAppStore((s) => s.layers);
  const snapshot = useAppStore((s) => s.latestSnapshot);

  const { nodes, edges } = useMemo(() => {
    const sizes = [inputSize, ...layers.map((l) => l.size)];
    const totalLayers = sizes.length;
    const layerHeights = sizes.map((s) => Math.min(s, 12));
    const nodes: Node[] = [];
    for (let layerIdx = 0; layerIdx < totalLayers; layerIdx++) {
      const size = sizes[layerIdx]!;
      const visible = layerHeights[layerIdx]!;
      const totalH = (visible - 1) * NODE_Y_GAP;
      for (let n = 0; n < visible; n++) {
        const isPlaceholder = n === visible - 1 && size > visible;
        const id = `${layerIdx}_${n}`;
        const label = isPlaceholder ? `+${size - visible + 1}` : '';
        const isInput = layerIdx === 0;
        const isOutput = layerIdx === totalLayers - 1;
        nodes.push({
          id,
          position: { x: layerIdx * NODE_X_GAP, y: -totalH / 2 + n * NODE_Y_GAP },
          data: { label },
          type: 'default',
          style: {
            width: 36,
            height: 36,
            borderRadius: 18,
            background: isInput
              ? 'rgba(56, 189, 248, 0.15)'
              : isOutput
                ? 'rgba(232, 121, 249, 0.15)'
                : 'rgba(167, 139, 250, 0.12)',
            border: `1px solid ${isInput ? '#38bdf8' : isOutput ? '#e879f9' : '#a78bfa'}`,
            color: 'hsl(var(--foreground))',
            fontSize: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          },
        });
      }
    }

    const edges: Edge[] = [];
    for (let layerIdx = 0; layerIdx < totalLayers - 1; layerIdx++) {
      const fromVisible = layerHeights[layerIdx]!;
      const toVisible = layerHeights[layerIdx + 1]!;
      const W = snapshot?.layers[layerIdx]?.weights;
      for (let i = 0; i < fromVisible; i++) {
        for (let j = 0; j < toVisible; j++) {
          const w = W?.[i]?.[j] ?? 0;
          const mag = Math.min(1, Math.abs(w));
          const positive = w >= 0;
          edges.push({
            id: `e_${layerIdx}_${i}_${j}`,
            source: `${layerIdx}_${i}`,
            target: `${layerIdx + 1}_${j}`,
            style: {
              stroke: positive ? '#34d399' : '#f87171',
              strokeWidth: 0.5 + mag * 2.5,
              opacity: 0.25 + mag * 0.7,
            },
          });
        }
      }
    }
    return { nodes, edges };
  }, [inputSize, layers, snapshot]);

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        minZoom={0.4}
        maxZoom={2}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="hsl(var(--border))" />
      </ReactFlow>
    </div>
  );
}
