# Examples/tile

## mlp - Neural Network Coordinate-Based per Rappresentazione Immagini

**Fenomeno simulato:**
Addestra una rete neurale multilayer perceptron coordinate-based per predire colori RGB a posizioni input date, utilizzando positional encoding per migliorare rappresentazione contenuti high-frequency. Confronta implementazioni Warp vs PyTorch.

**Funzionalità Warp utilizzate:**
• `wp.Tape()` automatic differentiation - Backpropagation nativa GPU attraverso network layers
• `warp.optim.Adam` - Ottimizzatore Adam integrato per parameter updates GPU-ottimizzati
• `wp.tile_*` operations - Sfruttamento tensor cores per matrix operations high-performance
• CUDA graph capture - Pre-compilazione training epochs per eliminating CPU overhead
• `wp.float16` precision - Memory efficiency e throughput migliorati su hardware moderno
• Block-level parallelization - Gestione automatica thread blocks per batch processing

**Capacità chiave:**
Warp compete direttamente con PyTorch per workloads ML moderni mantenendo integrazione seamless con simulazioni fisiche. Le tile operations sfruttano hardware specializzato ottenendo performance comparable ai framework ML dedicati.

## nbody - Simulazione Gravitazionale N-Body con Tile Primitives

**Fenomeno simulato:**
Simula un problema gravitazionale N-Body utilizzando approccio all-pairs con primitive tile di Warp. Le particelle sono inizialmente distribuite su una sfera e hanno velocità iniziali che creano rotazione, simulando la formazione di strutture gravitazionali.

**Funzionalità Warp utilizzate:**
• `wp.tile_load()` - Caricamento efficiente di dati in tile per sfruttare memoria condivisa GPU
• `wp.constant()` - Definizione costanti compile-time per parametri fisici (DT, SOFTENING_SQ, TILE_SIZE)
• Block-level parallelization con `block_dim=TILE_SIZE` - Ottimizzazione accesso memoria tramite tiling
• `body_body_interaction()` - Funzione per calcolo forza gravitazionale tra coppie particelle con softening
• All-pairs interaction pattern - Ogni particella interagisce con tutte le altre per simulazione accurata
• Array swapping - Gestione doppio buffer per aggiornamenti posizione senza race conditions

**Capacità chiave:**
Warp gestisce automaticamente tile operations per sfruttare architetture GPU moderne con shared memory. L'approccio tiled permette simulazioni N-Body scalabili fino a decine di migliaia di particelle mantenendo accuracy fisica e performance real-time per applicazioni astrofisiche.

## walker - Training Quadruped Soft-Body Locomotion con Neural Networks

**Fenomeno simulato:**
Allena un quadrupede mesh tetraedrico a camminare tramite rete neurale fully-connected che converte 8 fasi temporali sinusoidali in attivazioni tetraedriche. Il modello soft-body viene simulato forward e valutato sulla momentum del centro di massa.

**Funzionalità Warp utilizzate:**
• `wp.sim` soft-body physics - Simulazione tetraedri deformabili con contact handling e ground interaction
• `wp.launch_tiled()` con tile operations - Matrix multiplication ottimizzata via tensor cores per neural network
• `warp.optim.Adam` - Ottimizzatore integrato per training parameters della rete neurale  
• `wp.Tape()` + CUDA graph capture - Automatic differentiation attraverso physics simulation con pre-compilazione
• Tetrahedron activation control - Neural outputs interpretati come muscle activations per soft-body locomotion

**Capacità chiave:**
Warp unifica seamlessly machine learning con simulazioni fisiche complesse, permettendo end-to-end training di sistemi robotici soft attraverso differenziazione automatica. Il framework gestisce automaticamente gradienti attraverso physics engines per applications neuro-meccaniche avanzate.

---