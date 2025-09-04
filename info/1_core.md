# Examples/core

## dem - Simulazione Particelle Coesive

**Fenomeno simulato:**
Simulazione DEM (Discrete Element Method) di particelle che interagiscono tramite forze di contatto, attrito e coesione, con caduta gravitazionale e adesione al terreno.

**Funzionalità Warp utilizzate:**
• `@wp.kernel`/`@wp.func` - Decoratori per funzioni GPU parallelizzate
• `wp.HashGrid` - Struttura dati spaziale per ricerca rapida dei vicini
• `wp.hash_grid_query()` - Query spaziale efficiente per particelle adiacenti  
• `wp.array()` - Arrays GPU per posizioni, velocità e forze
• `wp.ScopedCapture`/CUDA graphs - Ottimizzazione performance tramite pre-compilazione
• `wp.render.UsdRenderer` - Rendering diretto in formato USD

**Capacità chiave:**
Warp gestisce automaticamente il passaggio CPU-GPU dei dati e ottimizza l'esecuzione tramite CUDA graphs. L'hashing spaziale permette simulazioni scalabili con migliaia di particelle mantenendo performance real-time.

## fluid - Simulazione Fluidi 2D

**Fenomeno simulato:**
Implementa un risolutore "Stable Fluids" 2D per la dinamica computazionale dei fluidi, con advection semi-Lagrangiano, proiezione della pressione e forze gravitazionali su griglia regolare.

**Funzionalità Warp utilizzate:**
• `wp.array2d()` - Arrays bidimensionali per simulazione su griglia
• `wp.constant()` - Costanti compile-time per dimensioni griglia
• `@wp.func` - Funzioni helper per interpolazione bilineare e sampling
• Multiple `@wp.kernel` - Kernels specializzati (advection, divergence, pressure_solve)
• `wp.ScopedCapture` - Ottimizzazione iterazioni pressione tramite CUDA graphs

**Capacità chiave:**
Warp gestisce automaticamente le operazioni su griglia 2D complesse con alta efficienza GPU e interpolazione smooth. Il ciclo iterativo del risolutore di pressione viene pre-compilato per massime performance real-time.

## graph capture - Generazione Noise Procedurale

**Fenomeno simulato:**
Genera noise procedurale animato usando fractional Brownian motion (FBM) con multiple ottave e frequenze. La griglia di coordinate scorre nel tempo per creare effetti animati.

**Funzionalità Warp utilizzate:**
• `wp.ScopedCapture()` - Cattura sequenze di kernel launches in CUDA graphs
• `wp.capture_launch()` - Esecuzione ottimizzata di graphs pre-registrati  
• `wp.noise()` - Generazione rumore procedurale GPU-nativo
• `wp.rand_init()` - Inizializzazione stati random per deterministico seeding
• Multiple kernel launches - Loop di 16 ottave con frequenze/ampiezze variabili
• `wp.ScopedTimer()` - Profiling performance automatico

**Capacità chiave:**
CUDA graph capture elimina l'overhead di launch ripetuti, accelerando drasticamente sequenze di kernel identici. Warp fornisce primitive noise/random ottimizzate per GPU, ideali per generazione procedurale real-time.

## marching cubes - Estrazione Superfici da Campi SDF

**Fenomeno simulato:**
Utilizza l'algoritmo marching cubes per estrarre superfici iso-livello da campi di densità SDF (Signed Distance Fields). Genera primitive geometriche animate (box, torus) combinate con operazioni smooth blending.

**Funzionalità Warp utilizzate:**
• `wp.MarchingCubes()` - Classe built-in per estrazione automatica superfici mesh
• `wp.array3d()` - Arrays tridimensionali per rappresentazione campi volumetrici  
• Funzioni SDF `@wp.func` - Primitive geometriche (box, torus) e trasformazioni spaziali
• `wp.quat_*()` operations - Sistema quaternioni per rotazioni animate smooth
• `.surface()` method - Estrazione mesh triangolare da threshold iso-superficie

**Capacità chiave:**
Warp integra nativamente marching cubes GPU-ottimizzato che converte automaticamente campi volumetrici in mesh renderizzabili. Le funzioni SDF procedurali permettono geometrie complesse animate con blending matematico preciso.

## mesh - Simulazione PBD con Collisioni Mesh Deformanti

**Fenomeno simulato:**
Simulazione particellare PBD (Position Based Dynamics) con collisioni contro una mesh triangolare deformante. Le particelle cadono per gravità e collidono con un modello 3D (bunny) che si deforma dinamicamente usando animazione sinusoidale.

**Funzionalità Warp utilizzate:**
• `wp.Mesh()` - Classe per mesh triangolari con BVH (Bounding Volume Hierarchy) integrato
• `wp.mesh_query_point_sign_normal()` - Query spatial collision point-to-mesh ottimizzate
• `wp.mesh_eval_position()` - Calcolo posizioni precise sulla superficie triangolare
• `.refit()` method - Aggiornamento dinamico BVH dopo deformazioni mesh
• USD integration - Caricamento asset tramite libreria Pixar USD

**Capacità chiave:**
Warp gestisce automaticamente strutture dati spaziali BVH che si aggiornano efficientemente durante deformazioni mesh. Le query di collisione sfruttano hardware-acceleration per performance real-time anche con migliaia di particelle.

## nvdb - Simulazione Particelle con Campi SDF Volumetrici

**Fenomeno simulato:**
Simulazione particellare PBD con collisioni contro campi SDF (Signed Distance Field) in formato NanoVDB. Le particelle cadono per gravità e collidono con geometrie complesse rappresentate come volumi discretizzati.

**Funzionalità Warp utilizzate:**
• `wp.Volume.load_from_nvdb()` - Caricamento diretto file NanoVDB da Houdini/Blender
• `wp.volume_sample_f()` - Sampling trilineare valori SDF da griglia volumetrica
• `wp.volume_sample_grad_f()` - Campionamento simultaneo valore e gradiente SDF
• `wp.volume_world_to_index()` - Conversione coordinate mondo-griglia automatica
• Custom `volume_grad()` - Calcolo gradienti tramite differenze finite per normali superficie

**Capacità chiave:**
Warp integra nativamente il formato NanoVDB industriale, permettendo collisioni precise con geometrie arbitrariamente complesse. Il sampling hardware-accelerated mantiene performance real-time anche con decine di migliaia di particelle su volumi ad alta risoluzione.

## raycast - Ray Tracer con Mesh Triangolari

**Fenomeno simulato:**
Implementa un ray tracer di base che lancia raggi dalla camera attraverso ogni pixel per calcolare intersezioni con mesh triangolari. Renderizza usando le normali delle superfici come valori di colore per visualizzazione 3D.

**Funzionalità Warp utilizzate:**
• `wp.mesh_query_ray()` - Query ray-triangle intersection accelerata hardware  
• `wp.Mesh()` - Struttura dati mesh con BVH automatico per spatial queries
• USD integration - Caricamento diretto asset da pipeline Pixar USD
• Parallelizzazione pixel - Un kernel thread per pixel con coordinate automatiche
• `query.normal` - Accesso diretto ai dati geometrici (normali, posizioni) dell'intersezione

**Capacità chiave:**
Warp trasforma automaticamente mesh triangolari in strutture BVH ottimizzate per ray casting GPU. La parallelizzazione massiva permette rendering real-time di scene complesse senza gestione manuale dei thread o ottimizzazioni spaziali.

## raymarch - Renderer SDF con Ray Marching

**Fenomeno simulato:**
Implementa un renderer ray marching che utilizza Signed Distance Functions (SDF) per definire geometrie procedurali. Crea scene con primitive (sfere, box, piani) combinate tramite operazioni booleane, con illuminazione avanzata che include diffuse, specular, fresnel e soft shadows.

**Funzionalità Warp utilizzate:**
• `@wp.func` SDF primitives - Funzioni matematiche per generare geometrie procedurali (sfera, box, piano)
• Operazioni booleane SDF - Union, subtract, intersect per combinare primitive complesse
• Adaptive ray marching - Loop con step size dinamico basato sulla distanza SDF
• Gradient-based normals - Calcolo normali tramite differenze finite del gradiente SDF
• Advanced lighting model - Diffuse, specular, fresnel reflection e soft shadow calculation
• Gamma correction - Post-processing colore per output realistico

**Capacità chiave:**
Warp permette rendering procedurale complesso interamente su GPU senza geometrie esplicite. Il ray marching matematico genera dettagli infiniti con illuminazione physically-based, ideale per scene procedurali e effetti artistici.

## sample mesh - Campionamento Uniforme Superfici Mesh

**Fenomeno simulato:**
Campiona punti uniformemente distribuiti sulla superficie di una mesh triangolare utilizzando una Cumulative Distribution Function (CDF). Calcola aree dei triangoli per costruire distribuzione probabilistica proporzionale, generando punti casuali che rispettano la densità geometrica della superficie.

**Funzionalità Warp utilizzate:**
• `wp.mesh_eval_position()` - Valutazione posizioni superficie tramite coordinate baricentriche
• `wp.atomic_add()` - Operazioni atomiche GPU per somme parallele thread-safe
• `wp.lower_bound()` - Binary search ottimizzata per sampling da CDF
• `wp.randf()` - Generazione numeri random GPU con seeding deterministico per frame
• Barycentric coordinate sampling - Generazione punti uniformi all'interno triangoli

**Capacità chiave:**
Warp implementa nativamente tutte le primitive per sampling geometrico avanzato, incluse ricerche binarie e coordinate baricentriche GPU-ottimizzate. Il sistema gestisce parallelizzazione massiva mantenendo distribuzione statistica matematicamente corretta per applicazioni Monte Carlo.

## sph - Simulazione Fluidi con Smoothed Particle Hydrodynamics

**Fenomeno simulato:**
Simulazione fluidi SPH utilizzando kernels matematici per calcolare densità, forze di pressione e viscosità tra particelle. Implementa schema kick-drift per integrazione temporale con gravità, bounds collision e damping per comportamento fluido realistico.

**Funzionalità Warp utilizzate:**
• `wp.HashGrid` con `wp.hash_grid_query()` - Ricerca vicini spaziale per interazioni SPH
• `wp.hash_grid_point_id()` - Ordinamento thread per celle per accesso memoria ottimizzato
• Custom SPH kernels `@wp.func` - Density, pressure e viscous kernels matematici
• Kick-drift integration - Schema numerico separato per velocità/posizione update
• Multi-step simulation - Substeps multipli per stabilità numerica con small timesteps

**Capacità chiave:**
Warp permette implementazione diretta delle equazioni SPH matematiche complesse su GPU mantenendo performance real-time. L'hashing spaziale automatico gestisce efficiently migliaia di interazioni particella-particella per fluidi convincenti.

## torch - Ottimizzazione Differenziabile con PyTorch Integration

**Fenomeno simulato:**
Ottimizza la funzione di Rosenbrock non-convessa usando l'ottimizzatore Adam di PyTorch su particelle distribuite. Dimostra l'integrazione seamless tra Warp e PyTorch per differenziazione automatica e machine learning.

**Funzionalità Warp utilizzate:**
• `torch.autograd.Function` - Classe custom per integrare kernels Warp in computational graph PyTorch
• `wp.from_torch()` / `wp.to_torch()` - Conversioni automatiche tensor PyTorch ↔ arrays Warp
• `adjoint=True` - Differenziazione automatica backwards pass per gradient computation
• `wp.device_to_torch()` - Gestione consistente device GPU tra frameworks
• Rosenbrock `@wp.func` - Funzione matematica complessa valutata in parallelo su migliaia di punti

**Capacità chiave:**
Warp si integra nativamente con PyTorch mantenendo performance GPU ottimali e supporting automatic differentiation. Permette di incorporare computazioni high-performance custom direttamente in neural network training pipelines senza overhead significativo.

## wave - Simulazione Equazione delle Onde 2D

**Fenomeno simulato:**
Risolve l'equazione delle onde 2D con differenze finite su griglia regolare, simulando propagazione ondosa con collisioni contro una sfera mobile. Integra temporalmente usando schema a substeps multipli per stabilità numerica.

**Funzionalità Warp utilizzate:**
• Finite difference `laplacian()` - Operatore nabla quadrato per equazione onde differenziale
• `wave_solve` kernel - Integratore temporale esplicito per propagazione ondosa
• `wave_displace` - Forcing term sinusoidale per generazione onde da sfera mobile
• Grid indexing 2D - Gestione automatica coordinate (x,y) → linear array indexing
• `grid_update` - Conversione height field → mesh vertices per rendering dinamico

**Capacità chiave:**
Warp permette simulazioni PDE (Partial Differential Equations) complete su GPU con substeps automatici per accuracy/performance trade-off ottimale. L'aggiornamento real-time dei vertices mesh crea visualizzazioni fluide integrate con la fisica.

---