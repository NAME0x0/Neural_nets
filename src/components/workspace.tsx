'use client';

import { useEffect, useRef, useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ArchitectureTab } from '@/components/tabs/architecture-tab';
import { DatasetsTab } from '@/components/tabs/datasets-tab';
import { TrainingTab } from '@/components/tabs/training-tab';
import { VisualizationTab } from '@/components/tabs/visualization-tab';
import { AnalysisTab } from '@/components/tabs/analysis-tab';
import { useAppStore } from '@/lib/store/use-app-store';
import { createTrainingWorker, type WorkerClient } from '@/lib/workers/client';

export function Workspace() {
  const [tab, setTab] = useState('architecture');
  const workerRef = useRef<WorkerClient | null>(null);

  const setSnapshot = useAppStore((s) => s.setSnapshot);
  const pushMetrics = useAppStore((s) => s.pushMetrics);
  const setTraining = useAppStore((s) => s.setTraining);

  useEffect(() => {
    const client = createTrainingWorker();
    workerRef.current = client;
    const off = client.on((msg) => {
      switch (msg.type) {
        case 'metrics':
          if (Number.isFinite(msg.loss)) {
            pushMetrics({ epoch: msg.epoch, step: msg.step, loss: msg.loss, accuracy: msg.accuracy });
          }
          setSnapshot(msg.weights);
          break;
        case 'done':
        case 'paused':
          setTraining(false);
          break;
        case 'reset_done':
          setTraining(false);
          break;
        case 'error':
          // eslint-disable-next-line no-console
          console.error('worker error', msg.message);
          setTraining(false);
          break;
      }
    });
    return () => {
      off();
      client.terminate();
      workerRef.current = null;
    };
  }, [pushMetrics, setSnapshot, setTraining]);

  return (
    <Tabs value={tab} onValueChange={setTab} className="space-y-4">
      <TabsList className="grid w-full max-w-3xl grid-cols-5">
        <TabsTrigger value="architecture" data-tour="architecture">
          Architecture
        </TabsTrigger>
        <TabsTrigger value="datasets" data-tour="datasets">
          Datasets
        </TabsTrigger>
        <TabsTrigger value="training" data-tour="training">
          Training
        </TabsTrigger>
        <TabsTrigger value="visualization" data-tour="visualization">
          Visualization
        </TabsTrigger>
        <TabsTrigger value="analysis" data-tour="analysis">
          Analysis
        </TabsTrigger>
      </TabsList>

      <TabsContent value="architecture">
        <ArchitectureTab />
      </TabsContent>
      <TabsContent value="datasets">
        <DatasetsTab />
      </TabsContent>
      <TabsContent value="training">
        <TrainingTab worker={workerRef} />
      </TabsContent>
      <TabsContent value="visualization">
        <VisualizationTab worker={workerRef} />
      </TabsContent>
      <TabsContent value="analysis">
        <AnalysisTab />
      </TabsContent>
    </Tabs>
  );
}
