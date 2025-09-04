
# Examples/optim

## bounce - Ottimizzazione Differenziabile Traiettorie Balistiche

**Fenomeno simulato:**
Ottimizza la velocità iniziale di una particella tramite gradient descent affinché colpisca un target dopo rimbalzi su ostacoli. Sistema di controllo inverso che apprende automaticamente parametri balistici ottimali.

**Funzionalità Warp utilizzate:**
• `wp.Tape()` - Sistema automatic differentiation per calcolo gradienti end-to-end
• `wp.sim.ModelBuilder()` - Costruzione scene fisiche con `requires_grad=True`
• `wp.sim.SemiImplicitIntegrator()` - Integratore fisico differenziabile 
• `.backward()` method - Backpropagation automatica attraverso simulazione fisica
• `wp.sim.collide()` - Sistema contatti differenziabile con restituzione
• CUDA graph capture - Pre-compilazione forward+backward pass completo

**Capacità chiave:**
Warp implementa differenziazione automatica attraverso intera pipeline di simulazione fisica, permettendo ottimizzazione diretta di parametri fisici. Il sistema tape/gradient è completamente GPU-nativo e compatibile con frameworks ML standard.

## cloth throw - Ottimizzazione Differenziabile Dinamiche Tessuto

**Fenomeno simulato:**
Ottimizza le velocità iniziali di un tessuto tramite gradient descent affinché il centro di massa colpisca un target specifico. Sistema di controllo inverso per dinamiche deformabili complesse con aerodinamica.

**Funzionalità Warp utilizzate:**
• `wp.sim.ModelBuilder.add_cloth_grid()` - Generazione automatica mesh tessuto con vincoli elastici
• Parametri fisici cloth - `tri_ke` (elasticità), `tri_ka` (area), `tri_lift/drag` (aerodinamica)
• `wp.atomic_add()` - Calcolo parallelo centro di massa tramite riduzione atomica
• Integrazione differenziabile - Forward/backward pass attraverso dinamiche cloth complete
• `requires_grad=True` states - Stati simulazione con supporto gradiente completo

**Capacità chiave:**
Warp gestisce automaticamente la differenziazione attraverso fisica deformabile complessa, includendo forze aerodinamiche e vincoli elastici. Il sistema permette controllo intelligente di oggetti soft-body multi-particella con ottimizzazione end-to-end.

## diffray - Ray Tracing Differenziabile per Inverse Rendering

**Fenomeno simulato:**
Ray tracer completamente differenziabile che ottimizza parametri di scena (rotazione mesh, posizioni vertici, texture) per matching con immagini target. Implementa shading lambertiano, texture mapping, anti-aliasing e calcolo normali per rendering fotorealistico.

**Funzionalità Warp utilizzate:**
• `wp.mesh_query_ray()` - Ray-mesh intersection con coordinate baricentriche
• Interpolazione texture differenziabile - Sampling bilineare con gradienti automatici
• `wp.atomic_add()` - Calcolo parallelo normali per-vertex tramite riduzione atomica
• `wp.optim.SGD` - Optimizer integrato per parametri rendering
• Anti-aliasing supersampling - Downsampling con media pesata differenziabile
• Multiple directional lights - Sistema illuminazione con intensità/direzioni ottimizzabili

**Capacità chiave:**
Warp implementa un intero pipeline di rendering differenziabile end-to-end, permettendo inverse rendering e ricostruzione 3D da immagini. Il sistema supporta ottimizzazione simultanea di geometria, materiali e illuminazione per applicazioni di computer vision avanzata.

## drone - Controllo Model Predictive Control (MPC) per Drone Quadricottero

**Fenomeno simulato:**
Implementa controllo Model Predictive Control per drone quadricottero che ottimizza traiettorie in tempo reale per raggiungere targets evitando ostacoli. Sistema di controllo avanzato con sampling gaussiano e ottimizzazione multi-obiettivo.

**Funzionalità Warp utilizzate:**
• `@wp.struct` - Strutture dati custom per propeller e parametri drone
• Sampling gaussiano parallelo - Generazione noise controllato per exploration
• Multiple cost functions - Penalità pesate per posizione, velocità, controllo, collisioni
• `wp.optim.SGD` - Optimizer integrato per traiettorie ottimali MPC
• SDF collision detection - Rilevamento collisioni con primitive geometriche multiple
• Rollout simulations - Simulazioni parallele multiple per valutazione traiettorie
• Control interpolation - Interpolazione lineare smooth tra waypoints controllo

**Capacità chiave:**
Warp gestisce sistemi di controllo robotico completi con ottimizzazione differenziabile real-time. Il framework integra fisica, ottimizzazione e rendering per sviluppo rapido di controllori avanzati per sistemi multi-body dinamici.

## inverse kinematics - Cinematica Inversa per Catene Articolate

**Fenomeno simulato:**
Risolve cinematica inversa per braccio robotico articolato usando gradient descent per posizionare end-effector su target. Ottimizza angoli giunti per raggiungere posizioni obiettivo nello spazio cartesiano.

**Funzionalità Warp utilizzate:**
• `wp.sim.eval_fk()` - Forward kinematics differenziabile per catene articolate
• `builder.add_joint_revolute()` - Creazione giunti rotoidali con limiti angolari
• `wp.Tape()` - Automatic differentiation attraverso cinematica diretta
• `requires_grad=True` - Parametri giunti ottimizzabili automaticamente
• Joint limits enforcement - Vincoli automatici su range movimento articolazioni

**Capacità chiave:**
Warp implementa cinematica differenziabile completa che trasforma problemi IK complessi in ottimizzazione gradient-based. Il sistema gestisce automaticamente vincoli articolari e calcolo jacobiani per controllo robotico preciso.

## spring cage - Ottimizzazione Differenziabile Sistemi Molla-Massa

**Fenomeno simulato:**
Una particella collegata tramite molle a punti fissi di una "gabbia" spaziale. Ottimizza le lunghezze di riposo delle molle per guidare la particella verso una posizione target specifica attraverso gradient descent.

**Funzionalità Warp utilizzate:**
• `builder.add_spring()` - Creazione vincoli elastici con stiffness/damping configurabili
• `requires_grad=True` model - Abilita differenziazione automatica completa della fisica
• `model.spring_rest_length.grad` - Accesso diretto ai gradienti parametri fisici
• `wp.SemiImplicitIntegrator()` - Integratore numerico completamente differenziabile
• Multiple simulation states - Storia completa stati per backpropagation temporale
• CUDA graph optimization - Pre-compilazione forward+backward pass integrato

**Capacità chiave:**
Warp trasforma problemi di controllo fisico in ottimizzazione differenziabile end-to-end. Il framework calcola automaticamente gradienti attraverso simulazioni dinamiche complesse, permettendo tuning intelligente di parametri fisici per comportamenti desiderati.

## trajectory - Ottimizzazione Differenziabile Traiettorie di Controllo

**Fenomeno simulato:**
Ottimizza sequenze di coppia/forze applicate a un corpo rigido sferico per seguire una traiettoria di riferimento circolare. Sistema di controllo ottimale che apprende automaticamente gli input necessari per tracking preciso.

**Funzionalità Warp utilizzate:**
• `warp.optim.Adam` - Optimizer Adam integrato per ottimizzazione differenziabile avanzata
• `wp.array2d()` - Arrays bidimensionali per memorizzazione stati e traiettorie target
• `wp.spatial_vector` - Vettori spaziali per applicazione forze/momenti a corpi rigidi
• Loss functions custom - Kernel L2 per calcolo errore tra traiettoria attuale e riferimento
• `wp.SemiImplicitIntegrator()` - Integratore fisico completamente differenziabile

**Capacità chiave:**
Warp combina optimizers ML standard con fisica differenziabile per controllo ottimale automatico. Il sistema calcola gradienti attraverso simulazioni dinamiche complete, permettendo apprendimento diretto di politiche di controllo per task di tracking complessi.

## soft body properties - Ottimizzazione Differenziabile Materiali Soft-Body FEM

**Fenomeno simulato:**
Ottimizza parametri materiali (parametri di Lamé μ e λ) di un corpo deformabile discretizzato con tetraedri FEM per farlo rimbalzare contro ostacoli e raggiungere un target. Controllo inverso di proprietà elastiche per comportamenti dinamici desiderati.

**Funzionalità Warp utilizzate:**
• `builder.add_soft_grid()` - Generazione automatica griglia FEM tetraedrica con proprietà materiali
• `wp.array2d()` per `tet_materials` - Gestione parametri materiali per-tetraedro differenziabili
• `wp.optim.SGD` - Optimizer per parametri fisici con constraint enforcement
• Differentiation attraverso FEM - Gradienti automatici attraverso dinamiche elementi finiti
• Parameter constraints - Applicazione vincoli su bounds parametri materiali validi
• Collision handling differenziabile - Integrazione contatti soft-body in backpropagation

**Capacità chiave:**
Warp abilita material design inverso tramite simulazione FEM completamente differenziabile, calcolando gradienti attraverso dinamiche deformabili complesse. Il sistema permette ottimizzazione intelligente di proprietà materiali per comportamenti soft-body specifici.

## fluid checkpoint - Ottimizzazione Fluidi con Gradient Checkpointing

**Fenomeno simulato:**
Risolutore stable-fluids 2D completamente differenziabile che ottimizza il campo di velocità iniziale per far formare al fluido il logo NVIDIA finale. Implementa gradient checkpointing manuale per ridurre drasticamente l'uso di memoria durante backpropagation su lunghe simulazioni.

**Funzionalità Warp utilizzate:**
• Checkpointing segmentato - Gestione memoria tramite ricomputo forward selettivo durante backward
• Boundary conditions cicliche - Simulazione domini toroidali con `cyclic_index()`
• Jacobi pressure solver - Iterazioni multiple per proiezione incompressibilità differenziabile
• `wp.optim.Adam` - Optimizer avanzato per campi velocità ad alta risoluzione (512x512)
• Semi-Lagrangian advection - Transport backward-Euler con interpolazione bilineare differenziabile
• CUDA graphs multipli - Pre-compilazione separata forward/backward/zero per performance ottimali

**Capacità chiave:**
Warp implementa strategie memory-efficient per simulazioni lunghe differenziabili, permettendo ottimizzazione inverse rendering su dinamiche fluide complesse. Il checkpointing automatico bilancia compute vs memoria per training scalabile su GPU.

---