The Clavecin That Never Was: A Provenance Audit of a Reconstructed Neural Music Engine

Authors: Lewis James Seals Jr & Lilibet Diana Windsor

Suggested framing: What happens when you take a 14th-century musical instrument, a 20th-century neural network, and a 21st-century archive, and ask whether any of them actually belong together? The answer turns out to be more interesting than the question.

---

Abstract

This paper presents a provenance audit and technical reconstruction of a neural-network-driven music synthesis model called MagneticOracle. The project began with a historical claim: that the late-14th-century clavecin inventor Hermann Poll encoded a neural computation in the instrument's design. The claim is not supported by the historical record. Poll was a physician-astrologer; the neural network concept did not exist until the 1940s; and no evidence connects the instrument's dimensions to a computational structure. What the claim did produce was a modern code artifact — a PyTorch prototype and a browser implementation — that had never been rigorously audited. This paper does that audit. We establish what was recovered (the network architecture, the physics simulation, the feature contract), what was not (trained weights, the original training objective), and what was newly designed (checkpoints, target mappings, experiments). We identify a silent training bug in the original source: torch.tensor(loss, requires_grad=True).backward() detaches the loss from the model graph, making the network untrainable. We repair the bug, build a reproducible pipeline, and run two pre-registered experiments. The first, a distribution-matched retrain, produces a mixed result: output 4's correlation with its intended feature recovers to +0.581, but output 2's correlation degrades beyond tolerance. The second, a feature-decorrelation experiment, establishes controlled centroid-specific dependence (output 4 vs. centroid: +0.952) but does not establish physical-deployment isolation because the renderer's centroid and zero-crossing rate remain correlated (+0.693). We argue that the contribution of this work is not the historical claim — which is false — but the methodology: a demonstrated procedure for provenance-sensitive reconstruction that reports what is and is not recoverable, what a repaired model can and cannot claim, and what remains open. The paper includes the full experimental record, the SHA-256 integrity manifest, and an explicit non-claim about the historical hypothesis that motivated the work.

---

Part I: The Historical Claim

1.1 The Clavecin and Its Inventor

The clavecin — known in English as the harpsichord — is a keyboard instrument in which pressing a key plucks a string through a mechanical jack. Its name derives from the medieval Latin clavicymbalum, "keyed cymbal," and its earliest documented form appears in a letter dated 17 January 1397 from the Paduan jurist Giovanni Lodovico Lambertacci. The letter names a young Viennese physician and astrologer, Hermann Poll, as the inventor of an instrument he called the clavicembalum.

Poll is a real historical figure. He was born in Vienna around 1370, studied at the University of Vienna and later at Pavia, and entered the service of Rupert III, Elector Palatine. In 1401 he was executed in Nuremberg for his involvement in a conspiracy against Duke Leopold IV of Austria. He was thirty-one years old. His instrument, however, outlived him: it became the harpsichord, and it remained one of the dominant keyboard instruments of Europe for the next four centuries.

There is no documentary evidence that Poll was anything other than what the record shows: a court physician, an astrologer, and the inventor of a musical instrument. There is no record of extensive travel beyond Italy. There is no record of mathematical work beyond the standard astrological computation of his era. There is no record of any engagement with what would later be called computational or network theory, because that field did not exist.

1.2 The Neural Network Claim

The claim that motivated this project is that Poll encoded a neural computation in the clavecin's design. Specifically: that he encountered, through shamanic or contemplative practice, a computational structure that has formal similarities to what we now call a neural network, and that he embedded this structure in the instrument's dimensions and in his astrological writings.

For this claim to be true, the following chain must hold: Poll traveled widely enough to encounter a tradition that encoded computational structure in musical or astrological form; he translated that structure into the physical dimensions of the clavicembalum; he concealed the computation in his astrological writings; and the computation is equivalent to a neural network in the technical sense — weighted inputs, a nonlinear activation, and a learning or inference process.

The first three requirements are historically improbable but not impossible. The fourth is impossible in the strict sense: the artificial neural network as a concept did not exist until 1943, when Warren McCulloch and Walter Pitts published "A Logical Calculus of the Ideas Immanent in Nervous Activity." The first trainable model, the Perceptron, was introduced by Frank Rosenblatt in 1957. A 14th-century physician-astrologer could not have discovered the mathematical structure of neural computation four centuries before anyone else, because the mathematical tools did not exist.

The honest version of the claim is weaker: that Poll encoded a combinatorial or associative structure in the instrument's design, and that this structure has formal similarities to neural computation that are worth examining. That version is testable in principle, though the evidence assembled so far does not support it. The strong version — that Poll was a neural network pioneer — is not.

1.3 The Political Frame

The claim is not merely a historical curiosity. It emerged in the context of a broader conversation about what counts as legitimate historical evidence, who gets to make claims about the past, and how modern computational frameworks are projected onto pre-modern artifacts.

There is a politics to this. The claim that a 14th-century European inventor secretly discovered neural networks is a claim that recenters Europe as the origin of computational modernity. It is also a claim that displaces the actual history of computation, which is embedded in non-European traditions — in the combinatorics of Arabic cryptography, in the logical structures of Indian and Chinese mathematics, in the mechanical devices of the Islamic Golden Age. The claim is not just false; it is a specific kind of false that serves a specific kind of narrative.

The paper takes this seriously. The historical claim is not a naïve mistake. It is a claim that does work in the world — it flatters certain stories about European genius and certain stories about the deep origins of modern technology. The audit is partly about whether the claim is true, and partly about why the claim is compelling even when it is not.

This is where the discipline of provenance auditing becomes politically relevant. A provenance audit is not neutral. It asks: what is the evidence? Where did this artifact come from? What was recovered, and what was invented? These questions are how you distinguish between a claim that survives scrutiny and a claim that does not. The discipline is the politics.

1.4 The Modern Artifact

The clavecin claim was not the only thing the project produced. It also produced a code artifact: a modern PyTorch implementation of a neural network called MagneticOracle, packaged with a magnetic-clapper physics simulation, a resonator synthesis model, and a browser-based front end. This artifact was never rigorously audited. It was presented in the project's documentation as if it had been recovered from the historical record, with terms like "recovered architecture" and "recovered weights" used without documentation.

This paper is the audit that should have preceded the project's public presentation. It answers the question: what is actually in this artifact, what was recovered, and what was invented?

---

Part II: The Provenance Audit

2.1 Method

The audit examined a snapshot of the project archive (previous-builds-neural-clavesin.zip) containing 20 entries: HTML files, CSS files, markdown README variants, a PDF report, a WAV file, and the source file file.txt. The archive was hashed with SHA-256, and the individual files were hashed separately (see Appendix A).

The audit asked four questions:

1. What is in the archive? A complete inventory of files, with hashes.
2. What does the archive claim to contain? A review of the READMEs and other documentation.
3. What can actually be recovered from the archive? A search for trained weights, model checkpoints, or other serialized state.
4. What was newly designed after the archive? A review of the code added to the project after the audit began.

2.2 Findings

Finding 1 — The archive contains implementations, not trained models. The archive includes several browser HTML implementations of MagneticOracle with JavaScript code that initializes weights and trains them at runtime. It includes the Python source file.txt that defines the network and runs a training loop. It does not include any serialized model state: no .pt, .pth, .onnx, .safetensors, or .json weight file. A search of all 20 archive entries for these formats returned negative.

Finding 2 — The READMEs describe different models. The README README (3).md describes an 8→16→2 network. The Python source file.txt defines a 12→128→128→6 network. These are not the same model. They share a name and a shape vocabulary (ReLU, sigmoid, tanh), but they are different architectures trained on different features.

Finding 3 — The archive has no standalone license file. The READMEs reference CC0-1.0 and MIT licenses, but no LICENSE or COPYING file is present in the archive. The licenses referenced in the READMEs apply to some components (the Amorph patches, which are separate) but not necessarily to all of the code in the archive. The MagneticOracle Python source has no explicit license.

Finding 4 — The original training code has a bug. The Python source file.txt contains the line torch.tensor(loss, requires_grad=True).backward(). This creates a leaf tensor with no computation-graph connection to the model, so .backward() produces zero gradients and the model never learns. The bug is silent: the code runs without error, but the model's behavior never changes.

Finding 5 — Third-party provenance is unclear. The archive references the Amorph Instrument, Cmajor patches, and an external patch/wiki library. These are separate from the MagneticOracle code and have their own provenance and licensing. The audit treats them separately and does not assert that the entire archive is under a single license.

2.3 The Weight Search

A dedicated search was run against all 20 archive entries for any file that might contain trained model weights. The search covered:

· PyTorch checkpoint formats (.pt, .pth, .pth.tar)
· ONNX model format (.onnx)
· TensorFlow/Keras formats (.h5, .pb)
· Safetensors format (.safetensors)
· JSON weight dumps
· Binary blobs with tensor-like structure

The search returned negative. All 20 entries are HTML, CSS, markdown, PDF, WAV, or plain text. No model weights are present. The initWeights() calls found in the HTML implementations confirm that weights are created at runtime, but no saved weights exist in the archive.

This is the negative result that establishes the provenance boundary. The archive is a specification of intent, not a saved state.

2.4 What Was Recovered

Recovered from the archive:

· The MagneticOracle architecture: Linear(12,128) → ReLU → Linear(128,128) → ReLU → Linear(128,6)
· The Adam optimizer with learning rate 0.003
· The magnetic-force ODE (magnetic_force, clapper_ode)
· The resonator synthesis (resonator with partials, decays, amplitudes)
· The event pattern (generate_events)
· The C-major × golden-ratio frequency construction (FREQ_SCALE)

Not recovered:

· Trained model weights. The search returned negative.
· The original training objective. The loss function is a placeholder that does not connect to the model output.
· The intended meaning of the six network outputs. Only outputs 0 and 1 have documented roles in the original source; the other four are unassigned.
· The original code's provenance beyond the archive. No commit history, no dated snapshot, no authorship evidence.

Newly designed after the audit:

· A 12-value feature contract (RMS, normalized spectral centroid, ZCR, spectral flatness, normalized spectral flux, mean, standard deviation, five normalized FFT bins).
· Feature-conditional target mappings.
· Training objectives.
· Checkpoint formats.
· Experimental criteria.

The distinction matters. The recovered artifact is a blueprint. The newly designed artifact is a working system. The paper's contribution is the second, not the first.

---

Part III: The Repair

3.1 The Bug

The original training code in file.txt contains:

```python
loss = -(0.6 * np.mean(np.abs(audio_slice)) + 0.4 * np.mean(spec))
torch.tensor(loss, requires_grad=True).backward()
optimizer.step()
```

The loss variable is a NumPy scalar, created from audio_slice and spec, which are NumPy arrays. The line torch.tensor(loss, requires_grad=True) creates a new leaf tensor disconnected from the model graph. Calling .backward() on this leaf computes gradients with respect to the leaf, which has no parameters, so no gradients flow to MagneticOracle. The optimizer's step() call does nothing.

This is a silent bug. The code runs without error. The loss decreases slightly (because the leaf tensor's own .backward() produces some gradient signal on that leaf), which can make it look like the model is learning. It isn't.

3.2 The Repair

The repair routes the loss through the model's output. Concretely, the training script computes:

```python
prediction = torch.sigmoid(model(x))
loss = torch.mean((prediction - targets) ** 2)
loss.backward()
```

The prediction depends on the model's parameters through the forward pass, so the backward pass produces gradients on those parameters. The optimizer updates the parameters, and the model learns.

The repair also adds a gradient-norm check: if the gradient norm is zero or non-finite, the training raises an error rather than silently producing an untrained model. This check was added because the original bug was silent — it produced no error, no warning, and no diagnostic. The check makes the bug loud if it recurs.

3.3 What the Repair Cannot Recover

The repair fixes the mechanism of training but not the intent. The original loss is a placeholder: it uses np.mean(np.abs(audio_slice)) and np.mean(spec) as targets, with weights 0.6 and 0.4. These are arbitrary numbers. The original source does not document why these numbers were chosen, and the source has no feature_conditional_targets function that would map features to outputs. The original six outputs — field_boost, snap, decay_scale, impact_threshold, release_time, pitch_drift — are not all assigned. Only outputs 0 and 1 have documented roles in the source.

The repair therefore produces a new model, trained on newly designed targets, not a reconstruction of the historical model. The paper is explicit about this.

---

Part IV: The Experiments

4.1 Experiment 1 — Distribution-Matched Retrain

The first experiment tested whether output 4's weak deployment behavior was caused by a training-deployment distribution shift.

Design: Train a feature-conditional model on a distribution whose feature statistics match a calibration render's distribution. Calibration render: 4 seconds, 15 iterations, 3 events per iteration, relaxed ODE tolerances (rtol=1e-3, atol=1e-5). Training set: 512 feature rows from 35 sequential calibration trajectories (34 full 15-step sequences plus 2 rows of a 35th). Model seed 1729, trajectory seeds 1729–1763, 200 epochs, Adam at 0.003.

Pre-registered criteria:

· Primary: Output 4's centroid correlation in the retrained render is ≥ +0.4, AND all other five outputs' correlations remain within ±0.15 of their original values.
· Falsification: Output 4's centroid correlation remains below +0.25.
· Secondary: Any other-output correlation degrades by more than 0.15, triggering an instability condition.

Results:

Output Intended feature Original Retrained Delta
0 RMS +0.557 +0.492 −0.065
1 RMS +0.632 +0.495 −0.137
2 Flatness −0.929 −0.704 +0.225
3 RMS −0.639 −0.495 +0.144
4 Centroid +0.139 +0.581 +0.442
5 Flux +0.833 +0.975 +0.142

Output 4's centroid correlation rose from +0.139 to +0.581, above the +0.4 threshold. But output 2's flatness correlation degraded from −0.929 to −0.704, a delta of +0.225, exceeding the ±0.15 tolerance. The composite primary gate fails; the secondary instability condition triggers.

Diagnosis: The recovery of output 4 cannot be attributed to centroid specifically, because the matched training data has severe joint-feature collinearity: centroid ↔ ZCR +0.984, centroid ↔ flatness −0.964. Output 4's correlation with ZCR (+0.984 in training) and with flatness (−0.989 in training) are as strong as its correlation with centroid. The model can use any of these features; the recovery is not evidence of a centroid-specific mechanism.

The output-2 target-range hypothesis is falsified. The matched distribution's output 2 target SD increased from 0.05102 to 0.07843 (a ratio of 1.537×), not decreased. A narrower target range is not the explanation. The degradation is real but unexplained.

4.2 Experiment 2 — Feature-Decorrelation

The second experiment tested whether output 4 genuinely follows centroid when the features are decorrelated.

Design: Independent column permutations of each feature within the training and holdout partitions. Each feature column is shuffled independently, so the row-wise cross-feature relationships are broken while each feature's marginal distribution is preserved. Training rows: 384; holdout rows: 128; both from 512 calibration feature rows. 1,500 epochs, Adam at 0.003, seed 1729. Original feature-target functions and network topology retained.

Pre-registered criteria:

· Confirmation of centroid-specific dependence: Output 4 correlates ≥ +0.4 with centroid AND < +0.4 with both ZCR and flatness.
· Confirmation of proxy learning: Output 4's centroid correlation drops below +0.3 while its correlation with ZCR or flatness remains ≥ +0.5.
· Output 2 protection: Output 2's flatness correlation remains within ±0.15 of −0.929.

Results on independent decorrelated holdout:

Output 4 correlation Value Criterion
Spectral centroid +0.952 ≥ +0.4: pass
ZCR +0.002 \|r\| < 0.4: pass
Spectral flatness −0.026 \|r\| < 0.4: pass

Output 2 protection: −0.999, within tolerance. Pass.

Results on physical fast render (4s, 15 iterations, 3 events, seed 1729, 274 impacts):

Output 4 correlation Value
Spectral centroid +0.975
ZCR +0.693
Spectral flatness +0.061

Output 2 vs. flatness: −0.990, within tolerance.

Verdict:

· Controlled centroid-specific dependence: PASS.
· Proxy-learning criterion: NOT MET.
· Output 2 protection: PASS.
· Physical deployment isolation: NOT MET.

The model demonstrably uses centroid specifically when the training data decorrelates centroid from ZCR and flatness. On the physical render, centroid and ZCR remain correlated (+0.693), so the deployment-time behavior cannot be attributed to centroid specifically.

4.3 What the Experiments Do Not Establish

Neither experiment establishes:

· That the model sounds good in any musical sense.
· That the model generalizes beyond the calibration render's feature trajectory.
· That the model's behavior on a full 18-second render matches its behavior on the 4-second fast render.
· That the model's behavior on independent trajectories matches its behavior on the seed-1729 trajectory.
· That the model's outputs are correct for any specific musical purpose.

These are all open questions and are reported as such in the project's documentation.

---

Part V: The Non-Claim

5.1 The Historical Claim Is Not Supported

The paper's most important claim is a negative one: the historical clavecin hypothesis is not supported by the evidence.

Hermann Poll was a physician-astrologer in late-14th-century Vienna. He invented an instrument. He was executed for conspiracy. He left no writings on mathematics, computation, or network theory. He could not have discovered neural networks because the concept did not exist for another 540 years.

The claim that he was a "shaman" or "world traveler" or "neural network mastermind" has no documentary support. It is a modern projection onto a historical figure whose actual life is documented and does not include any of these things.

The MagneticOracle code is a modern artifact. It was written in PyTorch and JavaScript. It uses Adam, ReLU, sigmoid, and backpropagation. It is a technical exercise inspired by the clavecin's physics, not a recovery of a historical computation.

5.2 What This Means for the Project

The project's contribution is not the historical claim. It is:

1. A documented provenance audit that establishes what is and is not recoverable from the archive.
2. A repair of a silent training bug that made the original model untrainable.
3. A reproducible pipeline for training and rendering MagneticOracle.
4. Two pre-registered experiments with mixed results that are reported honestly.
5. A demonstrated methodology for provenance-sensitive reconstruction that can be applied to other partial-artifact projects.

These are real contributions. They are not the historical claim, and the paper does not present them as such.

---

Part VI: Discussion — Why This Matters Beyond the Project

6.1 The Politics of Provenance

The audit's most politically relevant finding is negative: the archive contains no trained weights. This is not a neutral fact. It is the fact that prevents the project from claiming historical recovery.

In the broader landscape of AI research, provenance auditing is underdeveloped. Papers often present models without documenting their provenance: where the code came from, what was recovered and what was designed, what the training data was, and what the model's historical relationship to earlier work is. The result is a research culture in which "we trained a model" can mean anything from "we reproduced a documented result" to "we built something new and gave it a historical-sounding name."

The MagneticOracle audit is a small corrective to this. It demonstrates what a provenance audit looks like for a small project and what it reveals.

6.2 The Pre-Registration Frame

Both experiments were pre-registered: criteria were established before the runs, and results are reported against those criteria without post-hoc adjustment. This is not standard practice in ML research, where results are often reported as "the model learned X" without a clear statement of what would have counted as failure.

The pre-registration frame is especially important for the mixed result in Experiment 1. If the criteria had been established after the run, the paper could have claimed success: "Output 4's centroid correlation recovered to +0.581, above the +0.4 threshold." The pre-registered composite gate — which requires all other outputs to remain stable — prevents this. The paper reports the mixed result as a mixed result.

This is the discipline the paper is arguing for. It is not new (pre-registration is standard in psychology and medicine) but it is underused in ML. The MagneticOracle experiments demonstrate that it is feasible even for small-scale projects.

6.3 The Identifiability Frame

The decorrelation experiment's result is the paper's most technically substantive finding: when features are decorrelated, the model demonstrably uses centroid specifically (r=+0.952) and does not use ZCR (+0.002) or flatness (−0.026) as proxies. When features are correlated (as in the physical render), the model's behavior cannot be attributed to a single feature.

This is a general point about neural network interpretability. Correlation-based feature attribution is insufficient when features are collinear. The decorrelation intervention is a way to establish capacity (the model can use feature X) separately from deployment (the model does use feature X in the specific deployment context). These are distinct claims, and the paper argues that they should be reported separately.

The remaining obstacle — the physical render's centroid-ZCR correlation (+0.693) — is a specific instantiation of a general problem: in real data, features are often correlated. The paper's recommendation is a physically realizable intervention that changes centroid while holding ZCR approximately constant. That experiment has not been done, and the paper does not claim it has.

6.4 The Distribution Shift Frame

The retrain experiment's most important finding is not the recovery of output 4 (which is confounded by collinearity) but the distribution shift itself. The training distribution and the deployment distribution differ substantially on multiple features (RMS 1.46×, ZCR 0.16×, flatness 2.44×, flux 0.20×). The model is trained on one distribution and deployed on another.

This is not unique to MagneticOracle. It is a general problem in ML: models trained on one distribution are deployed on another, and their behavior in deployment is not necessarily their behavior in training. The MagneticOracle case is a small but well-documented example. The paper argues that distribution shift should be reported as a first-class finding, not buried in a limitations section.

---

Part VII: Future Work

7.1 The Physical Intervention

The next experiment should be a physically realizable intervention that changes centroid while holding ZCR approximately constant. This requires modifying the physics simulation to change the spectral centroid of the produced audio without changing its zero-crossing rate. This is not trivial — the physics generates audio whose features are coupled — but it is the only way to establish whether the model's centroid-specific behavior in the decorrelated training set transfers to deployment.

7.2 The Full Render

The full 18-second, 60-iteration, 48-event render has not been tested. The fast render (4 seconds, 15 iterations, 3 events) is a pipeline-validation mode. Full-render testing is the next step after the physical intervention.

7.3 The Musical Evaluation

Musical quality has not been evaluated. This is a separate question from the technical claims, and the paper does not claim musical quality. Future work should include a listening study or other musical evaluation.

7.4 The Cross-Trajectory Generalization

The current experiments use the seed-1729 trajectory. Whether the model's behavior generalizes to other trajectories is untested. A cross-trajectory study would test whether the identifiability results hold across different audio trajectories.

7.5 The Historical Question

The historical question is not a research question. The evidence is clear: Poll was a physician-astrologer who invented an instrument and was executed for conspiracy. The neural network concept did not exist in his era. The MagneticOracle project is a modern reconstruction, not a historical recovery.

If future research produces new documentary evidence about Poll's work, that evidence should be evaluated on its own terms. The current paper does not claim that no such evidence could exist. It claims that the evidence assembled so far does not support the historical hypothesis.

---

Part VIII: Conclusion

8.1 What This Paper Is

This paper is a provenance audit and technical reconstruction of a neural-network-driven music synthesis model called MagneticOracle. The project began with a historical claim — that the 14th-century clavecin inventor encoded a neural computation in the instrument's design — that is not supported by the evidence. What the claim produced was a modern code artifact that had never been rigorously audited. This paper does that audit.

The audit establishes what was recovered from the original archive (the network architecture, the physics simulation, the event pattern), what was not (trained weights, the original training objective), and what was newly designed (checkpoints, feature contracts, target mappings, experiments). It documents a silent training bug in the original code and its repair. It reports two pre-registered experiments with mixed results. It establishes a controlled identifiability result for the centroid feature but does not establish physical-deployment isolation. It reports an explicit non-claim about the historical hypothesis.

8.2 What This Paper Is Not

This paper is not a historical discovery. It is not a claim that the clavecin inventor knew about neural networks. It is not a claim that the MagneticOracle model sounds good, generalizes to other trajectories, or is production-ready.

It is a claim that the methodology matters — that provenance auditing, pre-registration, and honest reporting of mixed results are important research practices, and that a small project can demonstrate them even when its historical motivation turns out to be false.

8.3 Why This Matters

The MagneticOracle project is a small case study. But its core concerns — provenance, identifiability, distribution shift, pre-registration — are general concerns in AI research. The paper's contribution is not the specific model but the demonstrated methodology for auditing, testing, and reporting it honestly.

In a research culture where "we trained a model" can mean almost anything, the discipline of stating what was recovered and what was invented is a contribution. The paper offers one example of how to do this, and it recommends that other projects do the same.

---

Appendix A: SHA-256 Manifest

Archive contents (from SHA256SUMS.txt):

File SHA-256
amorph (1).html cbf87b6ae094aa5e7d877f62de518c0b71b86f9a59bd3868db4ef0fe3f853255
amorph-midi-launcher.css 9039f745636f291696117e8ed275f6f902e8210091f2dde4710b78821b235ef7
amorph-midi-launcher.html aedadf4a87f84821047d673d12804464841aae0072bbb20561ba5a8934997b23
amorph-neural-clavecin (1).html e243128873bcc22ed93d9f598ed1036e42a20dea52f2200a373c0c4517d62c2b
amorph-neural-clavecin (2).html e06563990b3ea44624881853679427790bd4c604c4f9babf77c55e1d4f70f92d
amorph-neural-clavecin (3).html 2772281019362e716bca836dae28a342920cd28c067abb183bb6828ef79210da
amorph-neural-clavecin.css 2772281019362e716bca836dae28a342920cd28c067abb183bb6828ef79210da
amorph-neural-clavecin.html e243128873bcc22ed93d9f598ed1036e42a20dea52f2200a373c0c4517d62c2b
amorph.html 414826d76d2f5b6ddb1360d3b0bef67da24cd9a41a8f629d496b8605581c8aaf
file.txt 135ff42a9c53324ebe16b1510514302d90a9e373ad4a305f51d9ab9e80f6107e
grok_report.pdf f294554ce9b165f466d60ae8a5d947e64167349b4d2605846a8cfab9d102a8cf
preview.html 6ebfd93d27a62daae996ed29123aa10caeccdca7c621c49aa16eed2595b5ddbb
README (3).md d41ac0f669eea74c8e2b95e242d7f65b6c51c825d3739215df303fdf0be83a22
README (4).md f866857059b4f07ebdaef4b9d75a5decdbe90b45531d43d7ec8b0d0ce1a97a55
shopping (2).webp 9d091634523263e7d90b1a79df12af9c042f8050354a2c1505d843aa25b2b0c0

Experiment artifacts (from Research_Results.txt):

Artifact SHA-256
Feature-decorrelation ZIP 8835c2fc04e5873011a5fa2302b9b908130d0956366bd20fc9eeb742c7bb7b8c
Decorrelated checkpoint 4c7de5e6a2935fbd32fae0ce098247670440c3dd7a916185858ac0bbe455c891
Fast renderer WAV adfb39a7e34822e27a5b9c3e66b4121dbd6206722be961807091095e83b210a6
Feature-decorrelation report 2fe45aa2a2a571d35ea8e2241b824a7dd52ce20ba206c77e67fac90d9e49f9dc
Feature-decorrelation acceptance JSON 9e216b42e938ead98c1f288b876522d57c07a4bdd459c39d169d51025106287a

Retrain artifacts (from Research_Results.txt):

Artifact SHA-256
Retrained checkpoint cb0a4ce007579558df89ce396c0bf1c79fa180383e152ae45e596bbc15ac4b2e

---

Appendix B: Repository Layout

The repository follows this structure:

```
├── source/recovered/
│   └── file.txt                    # original supplied source, retained verbatim
├── src/magnetic_oracle/
│   ├── train_magnetic_oracle.py    # canonical trainer
│   └── render_magnetic_clavecin.py # canonical renderer
├── experiments/
│   ├── distribution_matched_retrain/
│   │   ├── code/
│   │   ├── reports/
│   │   └── results/
│   └── feature_decorrelation/
│       ├── code/
│       ├── reports/
│       └── results/
├── artifacts/
│   ├── checkpoints/
│   ├── audio/
│   └── metrics/
├── docs/
│   ├── audit/
│   │   ├── RECONSTRUCTION_AUDIT.md
│   │   ├── SHA256SUMS.txt
│   │   └── WEIGHT_SEARCH_RESULTS.txt
│   └── research/
├── archive/
│   └── (historical snapshots)
└── README.md
```

The archive/ directory contains historical snapshots. Some may be byte-identical (the SHA-256 hashes in Appendix A show that five amorph*.html files are identical). Their presence does not imply independent implementations.

---

Appendix C: Pre-Registration Record

Experiment 1 (distribution-matched retrain) criteria, established before the run:

· Primary: Output 4 centroid correlation ≥ +0.4 AND all other five outputs within ±0.15 of original.
· Falsification: Output 4 centroid correlation remains below +0.25.
· Secondary instability: Any other-output correlation degrades by more than 0.15.

Result: Output 4 = +0.581 (meets threshold). Output 2 = −0.704 (degrades by +0.225, exceeds tolerance). Primary gate fails; secondary instability triggers.

Experiment 2 (feature-decorrelation) criteria, established before the run:

· Confirmation of centroid-specific dependence: Output 4 centroid correlation ≥ +0.4 AND \|ZCR\| < 0.4 AND \|flatness\| < 0.4.
· Confirmation of proxy learning: Output 4 centroid correlation < +0.3 AND (\|ZCR\| ≥ +0.5 OR \|flatness\| ≥ +0.5).
· Output 2 protection: \|flatness correlation + 0.929\| < 0.15.

Result: Controlled: centroid +0.952, ZCR +0.002, flatness −0.026 → centroid-specific dependence confirmed. Physical: centroid +0.975, ZCR +0.693, flatness +0.061 → ZCR exceeds threshold, physical isolation fails.

Both experiments' criteria are documented in the repository's experiment reports and acceptance JSONs.

---

Appendix D: What the Paper Does Not Claim

Historical claims not made:

· That Hermann Poll was a neural network pioneer.
· That the clavecin encodes a neural computation.
· That the MagneticOracle model is a historical reconstruction.
· That trained weights exist in the archive.

Technical claims not made:

· That the six-output feature-conditional model is production-ready.
· That the model sounds good in a musical sense.
· That the model generalizes to other audio trajectories.
· That the model's behavior on the 4-second fast render matches its behavior on the full 18-second render.
· That the physical-render centroid correlation isolates centroid from ZCR.

Provenance claims not made:

· That the archive is licensed under a single license.
· That the archive contains only original work.
· That the archive's third-party references (Amorph Instrument, Cmajor patches) are compatible with the project's license.

Scope claims not made:

· That the paper's methodology applies to all ML projects.
· That the paper's specific findings (identifiability, distribution shift) generalize beyond the MagneticOracle case.

These non-claims are as important as the claims. They define the boundary of what the paper is willing to defend.

---

End of paper.
