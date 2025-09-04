# Examples/sim

## cartpole - Simulazione Robotica Articolata URDF

**Fenomeno simulato:**
Simulazione dinamica multi-istanza di sistemi robotici articolati (cartpole) caricati da file URDF standard. Include cinematica diretta, vincoli sui joint e integrazione semi-implicita per stabilità numerica.

**Funzionalità Warp utilizzate:**
• `wp.sim.ModelBuilder()` - Costruzione modelli fisici con joint e constraint
• `wp.sim.parse_urdf()` - Parser integrato per file URDF robotici standard
• `wp.sim.SemiImplicitIntegrator()` - Integratore numerico stabile per corpi rigidi
• `wp.sim.eval_fk()` - Calcolo cinematica diretta automatico per catene articolate
• `wp.sim.render.SimRenderer()` - Rendering specializzato per visualizzazione robotica
• Multi-environment support - Simulazione parallela istanze multiple per RL/testing

**Capacità chiave:**
Warp fornisce una pipeline completa per robotica: caricamento URDF industriale, simulazione fisica accurata con vincoli, e rendering 3D integrato. Supporta parallelizzazione massiva per training reinforcement learning su centinaia di ambienti simultanei.

## cloth - Simulazione Tessuti con FEM e Collisioni

**Fenomeno simulato:**
Simulazione tessuti deformabili usando modelli FEM (Finite Element Method) con tre diversi integratori numerici. Il tessuto collide con corpi rigidi complessi e include vincoli, masse puntiformi e forze elastiche.

**Funzionalità Warp utilizzate:**
• `wp.sim.ModelBuilder.add_cloth_grid()` - Generazione automatica griglie tessuto FEM
• Multiple integrators - `SemiImplicitIntegrator`, `XPBDIntegrator`, `VBDIntegrator` per stabilità
• `wp.sim.collide()` - Sistema collisioni soft-body vs rigid-body ottimizzato
• `wp.sim.Mesh()` - Integrazione mesh USD per geometrie collisione complesse
• Parametri fisici configurabili - elasticità, damping, attrito per realismo materiali
• State swapping - Double buffering automatico per performance temporali

**Capacità chiave:**
Warp offre simulazione tessuti production-ready con integratori numerici multipli per trade-off stabilità/performance. Il sistema gestisce automaticamente topologie complesse, self-collisions e interazioni con rigid bodies usando algoritmi GPU-ottimizzati.

## granular - Simulazione Materiali Granulari Particellari

**Fenomeno simulato:**
Simulazione materiali granulari usando approccio particle-based con interazioni locali. Le particelle si comportano come sabbia/cereali con collisioni, attrito e proprietà di bulk material realistiche.

**Funzionalità Warp utilizzate:**
• `wp.sim.ModelBuilder.add_particle_grid()` - Generazione automatica griglie particellari 3D
• `self.model.particle_grid.build()` - Costruzione spatial hash per neighbor finding efficiente
• `wp.sim.SemiImplicitIntegrator()` - Integrazione temporale stabile per sistemi particellari
• Parametri fisici configurabili - `particle_kf`, `soft_contact_kd` per controllo comportamento granulare
• CUDA graph capture - Pre-compilazione loop simulazione per performance ottimali

**Capacità chiave:**
Warp gestisce automaticamente le interazioni tra migliaia di particelle tramite spatial hashing GPU-ottimizzato. Il framework permette simulazioni real-time di fenomeni granulari complessi (pile formation, flow dynamics) con controllo preciso delle proprietà fisiche del materiale.

## granular collision sdf - Simulazione Materiali Granulari Particellari

**Fenomeno simulato:**
Simulazione materiali granulari usando approccio particle-based con interazioni locali. Le particelle si comportano come sabbia/cereali con collisioni, attrito e proprietà di bulk material realistiche.

**Funzionalità Warp utilizzate:**
• `wp.sim.ModelBuilder.add_particle_grid()` - Generazione automatica griglie particellari 3D
• `self.model.particle_grid.build()` - Costruzione spatial hash per neighbor finding efficiente
• `wp.sim.SemiImplicitIntegrator()` - Integrazione temporale stabile per sistemi particellari
• Parametri fisici configurabili - `particle_kf`, `soft_contact_kd` per controllo comportamento granulare
• CUDA graph capture - Pre-compilazione loop simulazione per performance ottimali

**Capacità chiave:**
Warp gestisce automaticamente le interazioni tra migliaia di particelle tramite spatial hashing GPU-ottimizzato. Il framework permette simulazioni real-time di fenomeni granulari complessi (pile formation, flow dynamics) con controllo preciso delle proprietà fisiche del materiale.

## jacobian ik - Inverse Kinematics con Automatic Differentiation

**Fenomeno simulato:**
Risolve problemi di inverse kinematics robotica usando il metodo jacobiano trasposto. Calcola automaticamente matrici jacobiane per controllare end-effector di sistemi articolati verso posizioni target multiple in parallelo.

**Funzionalità Warp utilizzate:**
• `wp.Tape()` - Sistema automatic differentiation per calcolo gradienti simbolici
• `requires_grad=True` - Abilitazione gradient tracking su arrays e modelli fisici
• `wp.sim.eval_fk()` - Forward kinematics differenziabile per catene articolate
• `tape.backward()` / `tape.gradients` - Backpropagation automatica per derivate parziali
• Multi-environment parallelization - Calcolo jacobiani simultaneo su istanze multiple
• Gradient-based optimization loop - Update parameters via jacobian transpose method

**Capacità chiave:**
Warp integra automatic differentiation nativa nella simulazione fisica, permettendo ottimizzazione e controllo robotico real-time. I jacobiani vengono calcolati automaticamente via GPU senza approssimazioni numeriche, abilitando tecniche avanzate come reinforcement learning e optimal control.

## quadruped - Simulazione Robotica Quadrupede Multi-Environment

**Fenomeno simulato:**
Simulazione dinamica multi-istanza di robot quadrupedi articulati con controllo posizionale dei joint. Include cinematica floating-base, controllo PID integrato e distribuzione automatica ambienti per training parallelo.

**Funzionalità Warp utilizzate:**
• `wp.sim.FeatherstoneIntegrator()` - Integratore specializzato per dinamica robotica articolata
• `compute_env_offsets()` - Utility per distribuzione automatica ambienti multi-dimensionali
• `wp.sim.JOINT_MODE_TARGET_POSITION` - Controllo posizionale PID built-in per joint
• `floating=True` URDF parsing - Supporto base mobile per locomozione robotica
• Memory pooling + CUDA graphs - Ottimizzazioni avanzate per training RL massivo
• Multi-integrator support - XPBDIntegrator, SemiImplicitIntegrator, FeatherstoneIntegrator

**Capacità chiave:**
Warp offre simulazione robotica production-ready con integratori numerici specializzati per articolazioni. Il framework gestisce automaticamente centinaia di ambienti paralleli per reinforcement learning, mantenendo accuratezza fisica e performance real-time ottimali.

## rigid chain - Catene Articolate Multi-Joint

**Fenomeno simulato:**
Simulazione catene di corpi rigidi collegati da diversi tipi di joint meccanici. Dimostra comportamento dinamico di sistemi articolati con vincoli revoluti, sferici, universali, fissi e composti.

**Funzionalità Warp utilizzate:**
• `wp.sim.ModelBuilder.add_articulation()` - Costruzione sistemi articolati multi-body
• Multiple joint types - `JOINT_REVOLUTE`, `JOINT_BALL`, `JOINT_UNIVERSAL`, `JOINT_COMPOUND`, `JOINT_FIXED`
• `wp.sim.JointAxis()` - Definizione assi rotazione con limiti angolari configurabili
• `wp.sim.FeatherstoneIntegrator()` - Algoritmo Featherstone per dinamica articolata efficiente
• Joint limits/constraints - Controllo automatico limiti meccanici con rigidezza/damping
• Body/shape composition - Creazione geometrie complesse con proprietà inerziali

**Capacità chiave:**
Warp fornisce una libreria completa di joint meccanici per simulazione robotica e ingegneristica. Il sistema gestisce automaticamente vincoli complessi, limiti articolari e dinamica multi-body usando algoritmi numerici all'avanguardia per performance real-time.

## rigid contact - Collisioni Multi-Shape Corpi Rigidi

**Fenomeno simulato:**
Simulazione caduta libera e collisioni di corpi rigidi con geometrie diverse (box, sfere, capsule, mesh complesse). Include rotazioni iniziali e interazioni dinamiche con terreno e tra oggetti.

**Funzionalità Warp utilizzate:**
• Multiple shape primitives - `add_shape_box()`, `add_shape_sphere()`, `add_shape_capsule()`, `add_shape_mesh()`
• `wp.sim.collide()` - Sistema collisioni automatico rigid-body con broad/narrow phase
• USD mesh integration - Caricamento geometrie arbitrarie tramite `UsdGeom.Mesh()`
• `wp.sim.SemiImplicitIntegrator()` - Integrazione temporale stabile per dinamiche impulsive
• Parametri fisici configurabili - rigidezza (ke), damping (kd), attrito (kf) per realismo materiali

**Capacità chiave:**
Warp unifica primitive geometriche semplici e mesh complesse in un unico sistema di collisione GPU-ottimizzato. Gestisce automaticamente broad-phase detection e contact resolution per simulazioni multi-oggetto scalabili con centinaia di corpi simultanei.

## rigid force - Applicazione Forze Esterne Corpi Rigidi

**Fenomeno simulato:**
Simulazione corpo rigido con applicazione diretta di forze/torque esterni per produrre movimento controllato. Dimostra controllo dinamico preciso tramite forze programmatiche su sistema multi-body.

**Funzionalità Warp utilizzate:**
• `wp.sim.XPBDIntegrator()` - Integratore XPBD per dinamica stabile con forze impulsive
• `state.body_f.assign()` - Applicazione diretta forze/torque esterni su corpi specifici
• `SimRendererOpenGL()` - Rendering interattivo real-time per visualizzazione dinamica
• `wp.sim.collide()` - Sistema collisioni integrato con terreno e oggetti
• Force/torque control loop - Controllo continuo forze durante simulazione temporale

**Capacità chiave:**
Warp permette controllo diretto delle forze su ogni corpo rigido, abilitando implementazione di attuatori, motori e controlli robotici custom. Il sistema integra automaticamente forze esterne con vincoli fisici e collisioni per comportamenti realistici.

## rigid gyroscopic - Effetto Dzhanibekov e Instabilità Rotazionale

**Fenomeno simulato:**
Dimostra l'effetto Dzhanibekov dove corpi rigidi con distribuzione massa asimmetrica sviluppano tumbeling caotico nello spazio libero a causa di assi rotazionali instabili. Simula dinamica giroscopica complessa senza gravità.

**Funzionalità Warp utilizzate:**
• Multi-shape body composition - Combinazione box shapes con densità/posizioni diverse per inerzia asimmetrica
• `wp.sim.SemiImplicitIntegrator()` - Integrazione stabile per dinamica rotazionale complessa  
• `builder.body_qd` - Inizializzazione velocità angolari con perturbazioni precise
• `builder.gravity = 0.0` - Disabilitazione gravità per simulazione spazio libero
• `self.model.ground = False` - Rimozione vincoli terreno per movimento completamente libero

**Capacità chiave:**
Warp cattura accuratamente effetti fisici sottili come instabilità giroscopiche e precessioni, dimostrando precisione numerica per fenomeni dinamici complessi. Il sistema gestisce automaticamente tensori inerzia multi-shape per simulazioni realistiche di meccanica rotazionale avanzata.

## rigid soft contact - Interazioni Rigid-Soft Body con FEM

**Fenomeno simulato:**
Simulazione collisioni ibride tra corpi rigidi (sfera) e strutture deformabili FEM (trave elastica). La sfera cade e interagisce con una griglia di elementi finiti che si deforma elasticamente secondo parametri materiali configurabili.

**Funzionalità Warp utilizzate:**
• `builder.add_soft_grid()` - Generazione automatica griglie FEM tetraedriche 3D
• Parametri elastici Lamé - `k_mu`, `k_lambda` per controllo deformazione materiali
• Hybrid rigid-soft collision - Sistema collisioni unified per interazioni miste
• `soft_contact_*` parameters - Rigidezza/damping/attrito contact soft-body configurabili
• `wp.sim.SemiImplicitIntegrator()` - Integrazione temporale stabile per sistemi ibridi

**Capacità chiave:**
Warp unifica simulazione rigid e soft-body in un framework coeso, gestendo automaticamente le complesse interazioni tra materiali rigidi e deformabili. Il sistema FEM nativo supporta parametri fisici realistici per simulazioni ingegneristiche accurate.

## soft body - Materiali Hyperelastici FEM con Torsione Controllata

**Fenomeno simulato:**
Simulazione materiali Neo-Hookean hyperelastici usando FEM tetrahedrico con torsione controllata di 180°. Include calcolo conservazione volume durante deformazioni estreme e controllo boundary conditions per validazione numerica.

**Funzionalità Warp utilizzate:**
• `builder.add_soft_grid()` con parametri Lamé - Modellazione hyperelastica Neo-Hookean precisa
• Custom kernel `twist_points()` - Controllo boundary conditions programmabile per test specifici
• `fix_top`/`fix_bottom` constraints - Vincoli selettivi per controllo deformazione
• Volume computation kernel - Calcolo conservazione volume tramite integrazione tetrahedrica
• `wp.transform` applications - Applicazione rotazioni/traslazioni controllate su subset particelle

**Capacità chiave:**
Warp supporta modelli costitutivi avanzati per materiali hyperelastici con controllo preciso delle condizioni al contorno. Il framework permette validazione numerica tramite test controllati di deformazione, mantenendo accuratezza fisica per simulazioni ingegneristiche.

## cloth self contact - Tessuti FEM con Self-Contact VBD

**Fenomeno simulato:**
Simulazione tessuto FEM con torsione controllata che dimostra la capacità dell'integratore VBD di gestire auto-collisioni complesse mantenendo il tessuto completamente intersection-free durante deformazioni estreme.

**Funzionalità Warp utilizzate:**
• `wp.sim.VBDIntegrator(handle_self_contact=True)` - Integratore specializzato per self-contact avanzato
• `builder.add_cloth_mesh()` - Creazione cloth da mesh USD con parametri fisici configurabili
• Custom rotation kernels - `initialize_rotation()`, `apply_rotation()` per controllo boundary conditions
• `PARTICLE_FLAG_ACTIVE` manipulation - Controllo flags particelle per vincoli selettivi
• `integrator.rebuild_bvh()` - Ricostruzione periodica BVH per mantenere query efficiency
• Self-contact parameters - `soft_contact_radius`, `soft_contact_margin` per detection precisa

**Capacità chiave:**
Il VBDIntegrator di Warp risolve automaticamente self-contacts topologicamente complessi senza intersection artifacts, cruciale per tessuti durante deformazioni estreme. Il sistema gestisce intelligentemente la ricostruzione BVH per mantenere performance real-time anche con geometrie altamente deformate.

---