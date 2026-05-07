import type { WorkerInbound, WorkerOutbound } from './protocol';

export type WorkerClient = {
  post: (msg: WorkerInbound, transfer?: Transferable[]) => void;
  on: (handler: (msg: WorkerOutbound) => void) => () => void;
  terminate: () => void;
};

export function createTrainingWorker(): WorkerClient {
  const worker = new Worker(new URL('./training.worker.ts', import.meta.url), { type: 'module' });
  const handlers = new Set<(msg: WorkerOutbound) => void>();
  worker.onmessage = (ev: MessageEvent<WorkerOutbound>) => {
    handlers.forEach((h) => h(ev.data));
  };
  return {
    post(msg, transfer) {
      if (transfer && transfer.length) worker.postMessage(msg, transfer);
      else worker.postMessage(msg);
    },
    on(handler) {
      handlers.add(handler);
      return () => handlers.delete(handler);
    },
    terminate() {
      handlers.clear();
      worker.terminate();
    },
  };
}
