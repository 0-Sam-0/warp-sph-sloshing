# Examples/fem

## diffusion 3d - Risoluzione PDE con Metodi Elementi Finiti

**Fenomeno simulato:**
Risolve l'equazione di diffusione 3D "νΔu = 1" con condizioni al contorno miste (Neumann sui lati orizzontali, Dirichlet sugli altri) utilizzando metodi agli elementi finiti su diverse topologie mesh.

**Funzionalità Warp utilizzate:**
• `warp.fem` module - Framework completo elementi finiti con integrazione automatica
• `@fem.integrand` - Decoratori per forme matematiche deboli (diffusion, boundary forms)
• Multi-mesh support - Grid3D, Tetmesh, Hexmesh, Nanogrid, Trimesh3D, Quadmesh3D
• `fem.integrate()` - Assemblaggio automatico matrici e vettori right-hand-side
• Boundary condition handling - Proiezione sistemi lineari e weak enforcement
• `fem_example_utils.bsr_cg()` - Solver iterativo Conjugate Gradient ottimizzato

**Capacità chiave:**
Warp.fem fornisce un framework FEM completo GPU-nativo che gestisce automaticamente discretizzazioni spaziali complesse e assemblaggio matriciale. Supporta multiple tipologie mesh e condizioni al contorno con performance scalabili per problemi PDE industriali.

## mixed elasticity - Elasticità Non Lineare con Metodi Misti FEM

**Fenomeno simulato:**
Risolve l'equazione di equilibrio elastico non lineare Neo-Hookiano "Div[d/dF Ψ(F(u))] = 0" usando formulazione mista con spazi funzionali separati per spostamenti e stress. Applica iterazioni Newton per gestire la non linearità del modello costitutivo.

**Funzionalità Warp utilizzate:**
• Mixed FEM spaces - Spazi funzionali separati per displacement (Serendipity) e stress tensors
• `@fem.integrand` Neo-Hookean - Forme bilineari per stress/strain energy e approssimazione Gauss-Newton  
• Newton iterations - Loop automatizzato con assemblaggio matriciale incrementale per non linearità
• Block diagonal operations - Inversione matrici massa tau e accoppiamento gradient/stress
• Multiple element types - Triangular, quadrilateral con basis functions adattive
• Area conservation tracking - Integrazione controllo vincoli incompressibilità

**Capacità chiave:**
Warp.fem gestisce formulazioni miste avanzate con multiple unknown fields e solver non lineari robusti. L'assemblaggio automatico delle forme Newton-Raphson permette simulazioni iperelastiche industriali con controllo preciso delle proprietà materiali.

## apic fluid - Simulazione Fluidi con Metodo Affine Particle-In-Cell

**Fenomeno simulato:**
Simulazione fluidi APIC (Affine Particle-In-Cell) che combina particelle Lagrangiane con griglia Euleriana adattiva NanoVDB. Include gravità, incompressibilità, collisioni SDF e trasferimento bidirezionale particelle-griglia per conservazione momento angolare.

**Funzionalità Warp utilizzate:**
• `wp.Volume.allocate_by_voxels()` - Griglia adattiva automatica basata su distribuzione particelle
• `fem.PicQuadrature` - Classe specializzata per quadrature particle-in-cell con measures
• Mixed FEM spaces - Q1 velocity/fraction, P0 pressure per incompressible flow
• `fem.interpolate()` APIC advection - Trasferimento velocità e gradienti griglia→particelle  
• Incompressibility solver - Divergence-free projection con Schur complement method
• Collision SDF integration - Boundary conditions tramite signed distance functions

**Capacità chiave:**
Warp unifica seamlessly metodi particellari e FEM con griglia adattiva che si ridimensiona automaticamente. Il framework gestisce thousands di particelle con fisica fluidi accurata mantenendo performance real-time per simulazioni production-quality.

## streamlines - Generazione Streamlines 3D con Tracing Campi Velocità

**Fenomeno simulato:**
Genera streamlines 3D tracciando attraverso campi di velocità incompressibili con condizioni al contorno miste (inflow, outflow, free-slip). Utilizza forward tracing con step fisso per visualizzare patterns di flusso fluido.

**Funzionalità Warp utilizzate:**
• `fem.lookup()` - Operatore per spatial lookup e interpolazione field values in domini arbitrari
• `fem.Subdomain` - Definizione subset elementi per boundary conditions differenziate
• Raviart-Thomas elements - Spazi funzionali RT1/P0 per velocity/pressure incompressible flow
• Streamline generation - `fem.interpolate()` con spawn points jittered e forward integration
• Multiple boundary types - Classification automatica sides per inflow/outflow/freeslip conditions

**Capacità chiave:**
Warp.fem integra seamlessly solvers PDE con post-processing visualization avanzato, permettendo field tracing efficiente su geometrie complesse. Il lookup operator mantiene accuracy spatial anche durante advection fuori dal dominio originale.

## distortion energy - Ottimizzazione Parametrizzazione Superficie 3D

**Fenomeno simulato:**
Minimizza la distorsione di parametrizzazione (u,v) di superfici 3D utilizzando energia Symmetric Dirichlet "E(F) = ½|F|² + |F⁻¹|²" con iterazioni Newton per ottimizzazione non lineare del gradiente di deformazione F.

**Funzionalità Warp utilizzate:**
• `@fem.integrand` energy forms - Gradient/Hessian Symmetric Dirichlet per ottimizzazione parametrizzazione
• `fem.grad()` - Calcolo gradienti deformazione e approssimazione Gauss-Newton hessiana  
• Newton iterations - Loop automatizzato con line-search implicito per energy minimization
• `make_deformed_geometry()` - Costruzione geometrie deformate da displacement fields
• Multi-mesh support - Triangular, quad, deformed geometries con spazi parametrici 2D/3D
• Checkerboard visualization - Pattern rendering per validazione distorsione parametrizzazione

**Capacità chiave:**
Warp.fem gestisce ottimizzazione energy-based avanzata per parametrizzazione geometrica, essenziale per texture mapping e mesh processing. Il framework automatizza derivate complesse mantenendo accuracy numerica per applicazioni computer graphics industriali.

## navier stokes - Equazioni Navier-Stokes 2D con Advection Semi-Lagrangiana

**Fenomeno simulato:**
Risolve le equazioni Navier-Stokes 2D incompressibili "Du/dt - νΔ(u) + ∇p = 0, ∇·u = 0" con velocity-Dirichlet boundary conditions e schema advection semi-Lagrangiano per stabilità numerica delle forzanti convettive.

**Funzionalità Warp utilizzate:**
• Mixed FEM Q(d)-Q(d-1) - Spazi funzionali stabili velocity/pressure per sistemi saddle-point
• `fem.lookup()` semi-Lagrangian - Backtracking convection con interpolazione accuracy preserving
• `fem.ImplicitField` - Definizione boundary conditions tramite funzioni implicite
• `SaddleSystem` solver - Assemblaggio/soluzione automatica sistemi accoppiati incompressibili
• Hard boundary projection - Enforcing Dirichlet conditions tramite constraint matrix

**Capacità chiave:**
Warp.fem fornisce framework CFD completo con metodi avanzati per flussi incompressibili, incluso semi-Lagrangian advection che elimina restrizioni CFL. La gestione automatica di saddle-point systems permette simulazioni fluidi stabili anche ad alti Reynolds numbers.

## burgers - PDE Burgers Non-Viscosa con Discontinuous Galerkin

**Fenomeno simulato:**
Risolve la PDE Burgers inviscida non-conservativa "∂u/∂t + (u·∇)u = 0" utilizzando metodo Discontinuous Galerkin con limitatore di pendenza minmod per gestire soluzioni discontinue e shock formation.

**Funzionalità Warp utilizzate:**
• Discontinuous Galerkin spaces - Spazi funzionali discontinui per capturing shock waves
• `fem.jump()` / `fem.average()` - Operatori per condizioni salto/media alle interfacce elementi
• Upwind transport form - Schema upwind con flussi numerici per stabilità iperbolica  
• Minmod slope limiter - Limitatore pendenza per prevenire oscillazioni spurie vicino discontinuità
• SSPRK3 integration - Strong Stability Preserving Runge-Kutta 3rd order per conservation
• `fem.lookup()` neighbor access - Accesso celle adiacenti per slope limiting calculations

**Capacità chiave:**
Warp.fem gestisce completamente metodi DG per equazioni iperboliche non lineari, includendo limitatori shock-capturing automatici. Il framework mantiene properties conservativi essenziali mantenendo stability anche per soluzioni con gradienti estremi e discontinuità.

## magnetostatics - Magnetostatica 3D con Elementi Nédélec H(curl)

**Fenomeno simulato:**
Risolve un problema magnetostatico 3D (bobina rame con corrente radiale attorno a nucleo ferro cilindrico) usando formulazione curl-curl e spazi funzionali H(curl)-conformi per le equazioni Maxwell statiche "1/μ∇×B + j = 0".

**Funzionalità Warp utilizzate:**
• `fem.ElementBasis.NEDELEC_FIRST_KIND` - Elementi Nédélec per H(curl) conformity elettromagnetica
• `fem.ImplicitField` con geometry deformation - Mappatura cube→cylinder con gradiente analitico
• Multi-material domains - Campi permeabilità μ differenziati (ferro, rame, vuoto) tramite funzioni implicite
• `curl_curl_form` - Formulazione variazionale curl-curl per campi vettoriali elettromagnetici
• `fem.curl()` operator - Operatore curl built-in per post-processing campo magnetico B
• Deformed geometry construction - Geometrie complesse da implicit field mappings

**Capacità chiave:**
Warp.fem gestisce completamente problemi elettromagnetici industriali con geometrie arbitrarie e proprietà materiali complesse. Gli elementi Nédélec garantiscono continuità tangenziale appropriata per campi vettoriali, essenziale per accuratezza nelle simulazioni Maxwell.

## adaptive grid - Griglie Adattive per Simulazioni CFD con Colliders Complessi

**Fenomeno simulato:**
Simulazione flussi incompressibili attorno a geometrie complesse utilizzando griglie adattive che aumentano automaticamente la risoluzione vicino ai boundary dei colliders. Gestisce t-junctions e discontinuità tra livelli di raffinamento multipli.

**Funzionalità Warp utilizzate:**
• `fem.adaptivity.adaptive_nanogrid_from_field()` - Generazione automatica griglie multi-livello da refinement fields
• `fem.ImplicitField` refinement function - Controllo zones raffinamento tramite SDF distance-based criteria  
• H(div)-conforming Raviart-Thomas - Elementi che conservano massa localmente per accuracy CFD
• T-junction handling - Gestione automatica discontinuità ai boundary tra resolution levels
• `side_divergence_form` - Correzioni flusso per transizioni risoluzione non-conforming
• NanoVDB collider integration - Geometrie collisione complesse da pipeline industriali

**Capacità chiave:**
Warp.fem automatizza completamente adaptive mesh refinement mantenendo conservation properties e accuracy numerica. Le griglie adattive permettono simulazioni CFD scalabili con computational cost ottimale concentrando risoluzione solo nelle zone critiche.

## nonconforming contact - Contatto Elastico tra Corpi Non-Conformi

**Fenomeno simulato:**
Risolve un problema di contatto no-slip tra due corpi elastici discretizzati separatamente con "Div[E:D(u)] = g". Utilizza schema iterativo staggered dove i corpi si influenzano reciprocamente tramite displacement Dirichlet BC e applied boundary stress.

**Funzionalità Warp utilizzate:**
• `fem.field.NonconformingField` - Accoppiamento campi tra geometrie discretizzate indipendentemente
• `fem.SymmetricTensorMapper(wp.mat22)` - Storage ottimizzato tensori simmetrici (3 DOF vs 4 DOF completi)
• Mixed displacement-stress formulation - Spazi Serendipity S_k per displacement, Q_{k-1}d per stress
• Staggered iterative coupling - Soluzione alternata corpi con coupling tramite boundary conditions
• Damped stress updates - Controllo stabilità numerica tramite relaxation parameters

**Capacità chiave:**
Warp.fem gestisce nativamente problemi multi-body contact con discretizzazioni non-conformi, essenziale per simulazioni meccaniche industriali. Il framework automatizza trasferimento informazioni tra geometrie separate mantenendo accuracy fisica dell'interfaccia di contatto.

## darcy level-set optimization - Ottimizzazione Topologica Darcy Flow 2D

**Fenomeno simulato:**
Ottimizzazione forma basata su level set per massimizzare flusso Darcy attraverso dominio 2D quadrato. Evolve implicit shape representation tramite gradiente adiuvante per ottimizzare permeabilità regioni materiale sotto vincoli volume costante.

**Funzionalità Warp utilizzate:**
• `wp.Tape()` automatic differentiation - Calcolo gradienti adiuvanti attraverso PDE solve complessi
• Level set method con sigmoid smoothing - Rappresentazione implicita interfacce materiale differenziabili
• `fem.ImplicitField` per material properties - Campi permeabilità smooth function del level set
• Semi-Lagrangian/DG advection - Multiple schemi per evoluzione level set (continua/discontinua)
• Implicit Function Theorem - Differenziazione attraverso linear system solve iterativi
• Volume constraint penalty - Enforcement vincoli tramite weighted loss combination
• Forward/backward optimization loop - Gestione completa ciclo ottimizzazione topology

**Capacità chiave:**
Warp.fem unifica seamlessly PDE solving con automatic differentiation per problemi ottimizzazione topologica industriali. Il framework gestisce automaticamente adjoint computation attraverso implicit solvers, permitendo shape optimization scalabile per applicazioni engineering realistiche.

## elastic shape optimization - Ottimizzazione Forma Trave Elastica 2D

**Fenomeno simulato:**
Ottimizzazione forma trave cantilever 2D per minimizzare norma quadratica stress field (compliance) tramite analisi elementi finiti e gradient-based optimization. Trave fissata a sinistra con carico costante destro, con vincoli volume e boundary conditions.

**Funzionalità Warp utilizzate:**
• `wp.Tape()` + Implicit Function Theorem - Differenziazione automatica attraverso linear system solves elastici
• `warp.optim.Adam` - Ottimizzatore gradient-based integrato per nodal position updates
• Hooke elasticity `@fem.integrand` - Formulazione stress-strain con parametri Lame per materiali realistici
• Deformed geometry construction - Support per triangular/quad/grid meshes con vertex position optimization
• Quality regularization terms - Penalizzazione elementi degenerati/invertiti per mesh stability senza remeshing
• Fixed vertex projectors - Enforcement boundary constraints durante shape optimization

**Capacità chiave:**
Warp.fem automatizza completamente shape optimization strutturale con automatic differentiation attraverso PDE solvers complessi. L'integrazione nativa con optimizers scalabili permette design optimization industriale mantenendo physics accuracy e mesh quality.