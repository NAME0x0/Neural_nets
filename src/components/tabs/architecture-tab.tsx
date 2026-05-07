'use client';

import { Plus, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { HelpHint } from '@/components/ui/tooltip';
import { ACTIVATION_LIST } from '@/lib/nn/activations';
import { LOSS_LIST } from '@/lib/nn/losses';
import { useAppStore, newLayerId, type LayerSpec } from '@/lib/store/use-app-store';
import { NetworkGraph } from '@/components/network-graph';
import type { ActivationName, LossName, OptimizerName } from '@/lib/nn/types';

const OPTIMIZERS: OptimizerName[] = ['sgd', 'momentum', 'adam'];

export function ArchitectureTab() {
  const inputSize = useAppStore((s) => s.inputSize);
  const layers = useAppStore((s) => s.layers);
  const loss = useAppStore((s) => s.loss);
  const optimizer = useAppStore((s) => s.optimizer);
  const setInputSize = useAppStore((s) => s.setInputSize);
  const setLayers = useAppStore((s) => s.setLayers);
  const setLoss = useAppStore((s) => s.setLoss);
  const setOptimizer = useAppStore((s) => s.setOptimizer);

  const updateLayer = (id: string, patch: Partial<LayerSpec>) =>
    setLayers(layers.map((l) => (l.id === id ? { ...l, ...patch } : l)));
  const removeLayer = (id: string) => setLayers(layers.filter((l) => l.id !== id));
  const addLayer = () =>
    setLayers([
      ...layers.slice(0, -1),
      { id: newLayerId(), size: 8, activation: 'relu' },
      ...layers.slice(-1),
    ]);

  return (
    <div className="grid gap-4 lg:grid-cols-[420px_1fr]">
      <Card>
        <CardHeader>
          <CardTitle>Architecture</CardTitle>
          <CardDescription>
            Stack layers like LEGO bricks. The last layer is your output.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="input-size" className="flex items-center gap-1.5">
                Input size{' '}
                <HelpHint>
                  Number of features per example. For XOR it's 2; for a 28×28 image it's 784.
                </HelpHint>
              </Label>
              <Input
                id="input-size"
                type="number"
                min={1}
                value={inputSize}
                onChange={(e) => setInputSize(Math.max(1, Number(e.target.value) || 1))}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="flex items-center gap-1.5">
                Loss{' '}
                <HelpHint>
                  How wrongness is measured. MSE for regression, BCE for yes/no, CCE for
                  multi-class.
                </HelpHint>
              </Label>
              <Select value={loss} onValueChange={(v) => setLoss(v as LossName)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LOSS_LIST.map((l) => (
                    <SelectItem key={l} value={l}>
                      {l}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="col-span-2 space-y-1.5">
              <Label className="flex items-center gap-1.5">
                Optimizer{' '}
                <HelpHint>
                  Algorithm that updates weights from gradients. Adam usually works out of the box.
                </HelpHint>
              </Label>
              <Select value={optimizer} onValueChange={(v) => setOptimizer(v as OptimizerName)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {OPTIMIZERS.map((o) => (
                    <SelectItem key={o} value={o}>
                      {o}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Layers</Label>
              <Button size="sm" variant="secondary" onClick={addLayer}>
                <Plus className="mr-1 h-4 w-4" /> Add hidden
              </Button>
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Activation</TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {layers.map((l, i) => (
                  <TableRow key={l.id}>
                    <TableCell className="text-muted-foreground">
                      {i === layers.length - 1 ? 'out' : `h${i + 1}`}
                    </TableCell>
                    <TableCell>
                      <Input
                        type="number"
                        min={1}
                        value={l.size}
                        onChange={(e) =>
                          updateLayer(l.id, { size: Math.max(1, Number(e.target.value) || 1) })
                        }
                        className="h-8 w-20"
                      />
                    </TableCell>
                    <TableCell>
                      <Select
                        value={l.activation}
                        onValueChange={(v) =>
                          updateLayer(l.id, { activation: v as ActivationName })
                        }
                      >
                        <SelectTrigger className="h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {ACTIVATION_LIST.map((a) => (
                            <SelectItem key={a} value={a}>
                              {a}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </TableCell>
                    <TableCell>
                      <Button
                        size="icon"
                        variant="ghost"
                        disabled={layers.length <= 1}
                        onClick={() => removeLayer(l.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle>Network preview</CardTitle>
          <CardDescription>
            Live render. Edge thickness scales with weight magnitude.
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[520px] p-0">
          <NetworkGraph />
        </CardContent>
      </Card>
    </div>
  );
}
