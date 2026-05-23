Main Process
├── subprocess: phase4_inference.py   (ML model, camera)
├── subprocess: prediction_bridge.py  (sends to UI)
├── thread: HealthMonitor             (watches both subprocesses)
└── thread: LogAggregator             (reads their stderr)




┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKGROUND SERVICE (Step 3)                          │
└─────────────────────────────────────────────────────────────────────────────┘

Main Service Process (background_service.py)
│
├─────────────────────────────────────────────────────────────────────────────
│
├─ [Subprocess 1] phase4_inference.py
│  ├─ Runs in separate process
│  ├─ Produces predictions to Queue
│  └─ Monitored for crashes/restart
│
├─ [Queue] Shared IPC Queue
│  ├─ Predictions flow from inference → bridge
│  ├─ Thread-safe, multi-process compatible
│  └─ Monitored for overflow/stall
│
├─ [Subprocess 2] prediction_bridge.py
│  ├─ Runs in separate process
│  ├─ Consumes predictions, outputs to UI
│  └─ Monitored for crashes/restart
│
├─ [Health Monitor Thread]
│  ├─ Check subprocess status every N seconds
│  ├─ Detect crashes and restart
│  ├─ Monitor queue health
│  └─ Log diagnostics
│
├─ [Log Aggregator Thread]
│  ├─ Collect logs from all subprocesses
│  ├─ Write to central log file
│  └─ Display in console
│
└─ [Signal Handler]
   ├─ Catch SIGINT (Ctrl+C)
   ├─ Catch SIGTERM (kill signal)
   └─ Graceful shutdown sequence
