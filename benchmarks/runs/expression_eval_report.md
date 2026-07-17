# Expression Head Eval: DeepFace vs Py-Feat Detectorv2 vs EmotiEffLib

Generated 2026-07-17T13:18:11+00:00

- Photos evaluated: 30
- **Final recommendation: KEEP DeepFace (no candidate is both cheaper and objectively AU-validated)**

## A. Cost table

| model | mean latency/photo | load time | RSS delta |
|---|---|---|---|
| MediaPipe blendshapes | 8.2 ms | 5.781s | 546.1 MB |
| DeepFace emotion | 1050.5 ms | 5.111s | 6552.9 MB |
| Py-Feat Detectorv2 | 2724.9 ms | 7.875s | 770.1 MB |
| EmotiEffLib enet_b0_8_va_mtl | 21.0 ms | 13.765s | 775.8 MB |

## B. VA coherence check

- DeepFace: 30 faces scattered across 6 distinct labels; dominant label `sad` covers only 30.0%.

| candidate | n | mean valence | std valence | mean arousal | std arousal | dominant quadrant | dominant quadrant share |
|---|---|---|---|---|---|---|---|
| Py-Feat | 30 | 0.0018 | 0.3288 | -0.075 | 0.1484 | neg_valence_neg_arousal | 46.7% |
| EmotiEffLib | 30 | 0.0677 | 0.2993 | 0.0434 | 0.1609 | pos_valence_pos_arousal | 46.7% |

> Py-Feat's VA quadrant clustering is **more coherent than DeepFace's categorical scatter** on this corpus.

## C. AU sanity cross-check (eyes-closed)

- n=30 face photos with all three signals
- **3-way agreement (EAR == blendshape == AU43): 63.3%**
- EAR vs AU43: 76.7%
- blendshape vs AU43: 73.3%
- EAR vs blendshape (existing baseline): 76.7%

## Rubric verdicts

### Py-Feat Detectorv2: KEEP
- speedup 0.39x (threshold 5.0x)
- RSS savings 5782.8 MB (threshold 2000 MB)
- AU43-vs-EAR agreement 76.7% (threshold 85%)

### EmotiEffLib enet_b0_8_va_mtl: KEEP
- speedup 50.02x (threshold 5.0x)
- RSS savings 5777.1 MB (threshold 2000 MB)
- no AU43/eyes-closed signal to objectively validate

## Licensing flags

- MIT (deepface itself); backend model weights vary by provider.
- MIT (py-feat code); face_multitask_v2 CHECKPOINT is research/non-commercial only — flagged, not resolved, by this eval.
- Apache-2.0 (library and enet_b0_8_va_mtl weights) — fully permissive.

## Unresolved (pending owner reverence labels)

- Accuracy-for-reverence cannot be measured: no public dataset labels solemnity/reverence, so owner expression labeling on this corpus is required before any candidate's categorical or dimensional output can be scored against ground truth rather than just internal coherence.
- The VA coherence check (section B) is dispersion evidence, not accuracy: tighter clustering is evidence a signal COULD express reverence coherently, not proof that it does.
- py-feat's Detectorv2 device='auto' carries a known upstream FIXME (mixed cpu/mps ops can produce NaNs on some Mac configurations); this run resolved to MPS and produced non-NaN output on all 30 photos, but that is not a guarantee on other hardware/driver combinations.

## Per-photo table

| photo | deepface | pyfeat emotion | pyfeat V | pyfeat A | pyfeat AU43-closed | emotiefflib emotion | emotiefflib V | emotiefflib A | EAR-closed | blendshape-closed |
|---|---|---|---|---|---|---|---|---|---|---|
| DSCF0166 52 Edited.jpg | fear | Neutral | 0.26680445671081543 | -0.08322490751743317 | True | Surprise | 0.20369848608970642 | 0.14354434609413147 | True | False |
| DSCF0179.JPG | neutral | Neutral | -0.1412828117609024 | 0.012755357660353184 | False | Sadness | 0.1125754714012146 | 0.10814169049263 | True | True |
| DSCF0180 57 Edited.jpg | neutral | Happy | 0.7228402495384216 | -0.01623380556702614 | False | Neutral | 0.03891148418188095 | -0.04836075380444527 | False | False |
| DSCF0180.JPG | fear | Happy | 0.7422457337379456 | -0.01983477734029293 | False | Neutral | 0.1956082582473755 | 0.17331086099147797 | False | False |
| DSCF0210.JPG | neutral | Neutral | 0.14500963687896729 | -0.0519261509180069 | False | Happiness | 0.8176509141921997 | 0.15262600779533386 | False | False |
| DSCF0213.JPG | sad | Neutral | 0.17718496918678284 | -0.1492171585559845 | False | Neutral | -0.29154306650161743 | 0.10102897882461548 | False | True |
| DSCF0218 68 Edited.jpg | neutral | Happy | 0.10253977030515671 | 0.07620680332183838 | False | Neutral | 0.1572760045528412 | 0.08517991006374359 | False | False |
| DSCF0218.JPG | neutral | Sad | -0.5855624675750732 | -0.4529806971549988 | False | Neutral | 0.14097657799720764 | 0.015780005604028702 | False | True |
| DSCF0226.JPG | sad | Neutral | -0.27074503898620605 | -0.008985009975731373 | True | Neutral | 0.09028539061546326 | 0.1472238451242447 | True | True |
| DSCF0244.JPG | fear | Sad | -0.5997826457023621 | -0.44805026054382324 | False | Sadness | -0.6405203938484192 | -0.5957825779914856 | False | False |
| DSCF0252.JPG | neutral | Neutral | -0.27476733922958374 | -0.03237703815102577 | False | Neutral | -0.17184144258499146 | 0.024343162775039673 | True | True |
| DSCF0261.JPG | fear | Neutral | -0.28082597255706787 | -0.2834765613079071 | False | Surprise | 0.2711014151573181 | 0.1788163036108017 | True | False |
| DSCF0264.JPG | angry | Neutral | -0.20283722877502441 | -0.005550476722419262 | False | Neutral | -0.3213002383708954 | 0.19567781686782837 | True | True |
| DSCF0354.JPG | neutral | Neutral | -0.10490665584802628 | -0.04148782789707184 | False | Neutral | -0.1735081672668457 | -0.027983207255601883 | True | False |
| DSCF0363.JPG | happy | Neutral | -0.28642189502716064 | 0.16012166440486908 | False | Happiness | 0.6340799331665039 | 0.14165383577346802 | False | False |
| DSCF0366.JPG | angry | Neutral | -0.2139684557914734 | -0.1843058466911316 | False | Disgust | 0.13575230538845062 | 0.08034219592809677 | False | False |
| DSCF0369.JPG | sad | Neutral | -0.08672889322042465 | -0.05263537913560867 | False | Surprise | 0.0876404419541359 | 0.12665726244449615 | True | False |
| DSCF0370 145 Edited.jpg | sad | Neutral | -0.11356810480356216 | -0.13263538479804993 | False | Surprise | -0.16469749808311462 | 0.0701378583908081 | False | False |
| DSCF0373.JPG | surprise | Neutral | 0.10898452252149582 | -0.04732941463589668 | False | Neutral | 0.007115669548511505 | -0.1219959408044815 | False | False |
| DSCF0377.JPG | happy | Happy | 0.4383927881717682 | -0.03965306282043457 | False | Contempt | 0.0559287890791893 | -0.1326919049024582 | False | False |
| DSCF0384.JPG | sad | Neutral | -0.07579323649406433 | -0.09133514761924744 | True | Surprise | -0.10170940309762955 | 0.19013454020023346 | True | True |
| DSCF0387.JPG | sad | Neutral | 0.03710313141345978 | -0.031816646456718445 | False | Neutral | -0.27094244956970215 | -0.10022591054439545 | False | False |
| DSCF0457.JPG | happy | Neutral | -0.2631453573703766 | -0.21928873658180237 | False | Sadness | 0.23177886009216309 | -0.13640792667865753 | True | True |
| DSCF0484.JPG | sad | Neutral | -0.03283432126045227 | 0.018723415210843086 | True | Neutral | 0.1450107991695404 | 0.04674477502703667 | True | False |
| DSCF0494.JPG | sad | Happy | 0.6529560089111328 | 0.2157478630542755 | False | Neutral | 0.18564388155937195 | 0.1830344796180725 | False | False |
| DSCF0588 181 Edited.jpg | fear | Neutral | -0.0914146825671196 | 0.16316482424736023 | False | Surprise | -0.266126811504364 | 0.19763365387916565 | False | False |
| DSCF1422.JPG | angry | Happy | 0.3998781144618988 | -0.051834844052791595 | True | Anger | -0.03473200649023056 | 0.11616581678390503 | True | True |
| DSCF1490.JPG | fear | Neutral | 0.0010748255299404263 | -0.18000848591327667 | False | Happiness | 0.7253948450088501 | -0.027935225516557693 | False | False |
| DSCF1491.JPG | sad | Neutral | -0.020627647638320923 | -0.0510985367000103 | False | Neutral | 0.08412070572376251 | -0.14337900280952454 | False | False |
| DSCF1543.JPG | neutral | Neutral | -0.09690871089696884 | -0.22117413580417633 | False | Neutral | 0.1461600959300995 | 0.15853285789489746 | False | False |
