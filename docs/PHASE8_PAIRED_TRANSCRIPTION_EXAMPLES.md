# Phase 8 — 18 paired transcription examples

These are **18 real frozen-test clips**, each shown with its dataset reference, zero-shot B0 prediction, and validation-selected full fine-tuned F1 prediction. The same audio clip is used for both predictions. Text is the saved NFD-normalized prediction/reference; no text is truncated. IDs, speaker metadata, audio, and provenance are omitted.

**Selection:** examples were chosen *after* the declared final test to show a deliberate spread of F1 clip CER (roughly 15–167%), duration (about 2–20 seconds), B0 behavior, and two cases where F1 is worse. This is an **illustrative stratified set**, not a random sample or a performance estimate. The primary result is still corpus CER on **all 683 test clips**: B0 **101.23214%**, F1 **44.57355%**. No model or decoding setting was selected using these examples.

CER is `(S + D + I) / reference Unicode code points × 100` after the declared NFD/whitespace policy. A clip CER above 100% is possible when insertions dominate. `S/D/I` counts come from the saved Phase 8 scorer and were checked here against an independent minimum-edit-distance calculation. These IPA-like references have not been adjudicated by a Nyishi expert; no English translations are asserted.

## Quick comparison

| Example | Duration | Reference code points | B0 CER | Final F1 CER | Change | B0 cap hit |
|---:|---:|---:|---:|---:|---:|---|
| 01 | 1.96 s | 31 | 35.48% | 16.13% | +19.35 pp | No |
| 02 | 9.73 s | 45 | 33.33% | 15.56% | +17.78 pp | No |
| 03 | 14.25 s | 93 | 51.61% | 19.35% | +32.26 pp | No |
| 04 | 3.17 s | 30 | 126.67% | 26.67% | +100.00 pp | No |
| 05 | 8.90 s | 98 | 60.20% | 39.80% | +20.41 pp | No |
| 06 | 9.80 s | 112 | 88.39% | 25.89% | +62.50 pp | No |
| 07 | 2.00 s | 22 | 95.45% | 54.55% | +40.91 pp | No |
| 08 | 7.95 s | 111 | 60.36% | 42.34% | +18.02 pp | No |
| 09 | 16.22 s | 122 | 76.23% | 41.80% | +34.43 pp | No |
| 10 | 20.36 s | 139 | 65.47% | 43.88% | +21.58 pp | No |
| 11 | 4.01 s | 44 | 36.36% | 77.27% | -40.91 pp | No |
| 12 | 11.98 s | 102 | 96.08% | 60.78% | +35.29 pp | No |
| 13 | 15.14 s | 126 | 94.44% | 61.90% | +32.54 pp | No |
| 14 | 2.66 s | 32 | 90.62% | 84.38% | +6.25 pp | No |
| 15 | 9.18 s | 66 | 84.85% | 80.30% | +4.55 pp | No |
| 16 | 15.27 s | 131 | 99.24% | 83.21% | +16.03 pp | No |
| 17 | 4.70 s | 27 | 59.26% | 166.67% | -107.41 pp | No |
| 18 | 16.03 s | 75 | 518.67% | 130.67% | +388.00 pp | Yes |

## Complete paired predictions

### Example 01 — 1.96 s; B0 35.48%, final F1 16.13%

**Dataset reference:** `ŋo so piɟikal diparment pori do`

**B0 zero-shot prediction:** `mozopizikel depamentso porido`

**Final fine-tuned prediction:** `ŋoso piɟikel diparmentso porido`

**Edit counts:** B0 S/D/I = 5/4/2; final F1 S/D/I = 1/2/2. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 02 — 9.73 s; B0 33.33%, final F1 15.56%

**Dataset reference:** `ane suilo ane pajilo dente teminko ale nikune`

**B0 zero-shot prediction:** `ne suʒilou anepoiʒilou dintɛ temiko alɛni kunɛ`

**Final fine-tuned prediction:** `ane suɨjilo o ane bajilo dente teminko aleni kune`

**Edit counts:** B0 S/D/I = 8/3/4; final F1 S/D/I = 1/1/5. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 insertions exceed its other edit types combined; extra output remains a visible failure mode.

### Example 03 — 14.25 s; B0 51.61%, final F1 19.35%

**Dataset reference:** `ɟun nam marcam am am biɟo nopa aldo ho marc epril meŋam meŋam meŋas rule ŋul ɟunam am am dido`

**B0 zero-shot prediction:** `dʒun maɾxam abam biʝonapaldo maɾx epɾelmẽɣa nẽɣa sɾolãgul dʒunam am ðjɾo`

**Final fine-tuned prediction:** `ɟunnam maːrcəm abam biɟunə paldo ho marʈ hepril meŋəm meŋəm meŋəs role ŋul ɟunəm am am dido`

**Edit counts:** B0 S/D/I = 28/19/1; final F1 S/D/I = 12/4/2. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 04 — 3.17 s; B0 126.67%, final F1 26.67%

**Dataset reference:** `ma ŋulo ho pagoa ŋo bende lama`

**B0 zero-shot prediction:** `maɪː˧˥mɔː˧˥loː˧˥hɔː˧˥ paːguaː˧˥muː˧˥ bjɛn˥˩˧tɛl maɪː˧˥`

**Final fine-tuned prediction:** `maːŋulo ho paɡoaːm ŋo bendrla maː`

**Edit counts:** B0 S/D/I = 14/0/24; final F1 S/D/I = 3/1/4. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 05 — 8.90 s; B0 60.20%, final F1 39.80%

**Dataset reference:** `ama dedub lod̪a lik bohel mulɡe pas moinaŋəm pas moinaŋəm mule pura aməm pel nalde pule amcerlendo`

**B0 zero-shot prediction:** `amɛdɛduɠlo daɫik bolhɛmuɫgɛ pasmɔjnɛᵑɛm pasmɔjnɛᵑɛlɛmɛlɛmɛlɛmɛlɛmɛl pɔblɛdko ɛmtɛɾlɛndo`

**Final fine-tuned prediction:** `amədə dublo likbol həm mulge pas moɪnaŋəm pas moɪnaŋəm pas moɪnaŋəm pijəl nardəpublɯk həmcir lendo`

**Edit counts:** B0 S/D/I = 48/11/0; final F1 S/D/I = 23/8/8. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 06 — 9.80 s; B0 88.39%, final F1 25.89%

**Dataset reference:** `mul sopa padɔm sopa masina madohəm mule sapa pandub mado hoɡe sinəme simɯ pardob nəma kalaɡe hai hara mananedodo`

**B0 zero-shot prediction:** `mulˌsaːpˌaːpˌaːdɐmˌsaːpˈama sinˌɛmˌaːdɐwˌhɐmulˈaːsˌaːpˈaːpˌaːnˌduːmˌaːdɐwˌhɔgɐ siːnˈnˌeːmˌeːsˌeːmˌɐlɐpˈɐɾnduːkˌ neːmˌeːkɐlɐmˌgɐ ɦɾɐmˈaːnˌaːlɐnˌeː dɔˈdɔːdɔː`

**Final fine-tuned prediction:** `mul səpa padəm səpa məsinəm mado həm mule səpa mandub mado hoɡe sineməm siməməl siməpərndob nəma kalam ɡahara manale dodo`

**Edit counts:** B0 S/D/I = 56/0/43; final F1 S/D/I = 14/3/12. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 07 — 2.00 s; B0 95.45%, final F1 54.55%

**Dataset reference:** `mah mule heb madu mule`

**B0 zero-shot prediction:** `mɑm mɪlɛhɛ biːd ʒyːdnumɑdɛ wɯl`

**Final fine-tuned prediction:** `maː mule habijet momado`

**Edit counts:** B0 S/D/I = 11/1/9; final F1 S/D/I = 11/0/1. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 08 — 7.95 s; B0 60.36%, final F1 42.34%

**Dataset reference:** `nokəm niŋe mulɡa text̪a mulu ɡinəm inaməm muluɡ denəm tinaməm noh tot̪ab apa ekdom mulpa pajalinco nampali ɟido`

**B0 zero-shot prediction:** `no kamɲiŋa mulgɑrtexta mululuginam unamamuluk dɛnam tɾinam amnotata ɓɑpɑ ɛtɗɔmulpɑjlɛntʃon nɑmɑlidʒɑm ɓɑlidʒɑm ɓɑpɑdɔ`

**Final fine-tuned prediction:** `nokam ɲiŋe mul gef tekʰ t̪aːb ŋulu ɡinəm unam unaməm denəm tenəm t̪anəməm no t̪at̪ab pa edum mumpajelinson nəm palicɔp`

**Edit counts:** B0 S/D/I = 35/13/19; final F1 S/D/I = 24/8/15. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 09 — 16.22 s; B0 76.23%, final F1 41.80%

**Dataset reference:** `ŋo e aːrt̪em cjɔgʌ mɔ paːs baːɟe saːre paːs baːɟe gɔrʌb laːjel ɔgmo hɔŋhoroŋo ɨd̪ɔ rɔŋo uɪlaːjl or naː laːjel mosɪ opjud̪o`

**B0 zero-shot prediction:** `ɡuɛ ɑɾkɛɴ tʃoka ɡuɛ pɑsɛbɑ ʄi sɑɾipa sɔɾɑɓɛl ɔɓɛl ɦoɛᵑɛɾɔᵑɛɛ ɦoɛ ɦoɔᵑɛɾɔᵑɛ ɦoɛ ɔnɑnɛlɛ mɔsi ɔpjɔbɔ`

**Final fine-tuned prediction:** `ŋɔ iː ɪ aːrtəm ɟjoge ŋo pas baɟi sari pasa baɟi korəblajel hoŋo hoŋo roŋo roŋo udo roŋo ulajil o nalajel mosi opijedo`

**Edit counts:** B0 S/D/I = 67/25/1; final F1 S/D/I = 28/14/9. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 10 — 20.36 s; B0 65.47%, final F1 43.88%

**Dataset reference:** `oge so esco debləpment oge so roŋo nampo e mane egrikalcaram oʃe mədot məto e mane he plantiʃon hugub məta e iskim iskim nalal məte saɟeʃən`

**B0 zero-shot prediction:** `oɠe soɛsʃo doblɛmɛnto ɔgɛso ɛ ɾɔmɔnɛm po iɛ manɛ eɣɾikɑl tʃɛɾɛm ɔmʃɛmɔdɔtɔ mɛto iɛ manɛ ɛhɛlɛnti sɔn hɔgumɛtɛlɛn hɑm manɛ ɪskɛm sɪskɛm nɑlɛl hɛmɛt kɑhɛsɑ sɪsɔn sɪtɔ`

**Final fine-tuned prediction:** `oge soesco sobes ɟo doblument hoge so roŋonam po iː mane egrikal cʰaram ocemodot mədok meʈo iː mane ehe plan tison hogu matailen ham mane iskim siskin nalel həb`

**Edit counts:** B0 S/D/I = 58/4/29; final F1 S/D/I = 16/12/33. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 insertions exceed its other edit types combined; extra output remains a visible failure mode.

### Example 11 — 4.01 s; B0 36.36%, final F1 77.27%

**Dataset reference:** `pebuari tonti por se tonti siks tək ɲokɯm am`

**B0 zero-shot prediction:** `tʃeboarɨ tontʃi poɾsiɛ tontʃi sɪkstak jokumam`

**Final fine-tuned prediction:** `peboəri t̪ot̪ʰɔt̪ʰiː pɔrse t̪o t̪ɔncɪ sɪsɪst̪a kʰɔnjokumən`

**Edit counts:** B0 S/D/I = 11/2/3; final F1 S/D/I = 16/2/16. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 makes more edits than B0 on this clip. This counterexample prevents treating the aggregate gain as universal.

### Example 12 — 11.98 s; B0 96.08%, final F1 60.78%

**Dataset reference:** `porilehi asamo bakma porlai malal eiɡo hora aʔkaləm pirnəm bareŋəm mara cenu pare porilai hoɡe cint̪ai`

**B0 zero-shot prediction:** `poriʎi bolgiŋa asama paqomadu poɾleʃtuʃiʋɔʋɔʋɔʋɔʋɔʋɛl ɛlgoʊɦɾɑ ʔɑtɯʋɔnɔɔm ʋɑɾɛgɛmʊʋɔɾɛgɛmʊ ʋɑɾɛgɔmɛrɔʃɛndu pɑɾɛporiʎɛlɔgɾui ʋɛnʃiʋɛnɛtʃiʋɛ`

**Final fine-tuned prediction:** `pori lehi bolniŋa asa pak ŋado por lehtut̪is maɡku ohab male el ko hara aʔi nampinəm bareŋəm cenup pori lai hoɡlui`

**Edit counts:** B0 S/D/I = 60/1/37; final F1 S/D/I = 16/17/29. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 13 — 15.14 s; B0 94.44%, final F1 61.90%

**Dataset reference:** `lokəl dubai ded̪am aiko alpa həm dekəm bulom mahəmso osai tomahəb d̪aika aico alaɡe dohəmho maledo lokəl hamo aicodeməm laɡido`

**B0 zero-shot prediction:** `lo kando waɗɛnɛmɑɗi ɛɗɛlɛlɛlɛl pɑlɛlɛl mɑɑ mɑɗɛ kɛmɑlumɛl ɛɑmɑ hɛlɛlɛl tɔ ɛsɑjɛtɛpɪk tɔɔmɑ hɑmɑ ɑl kɛmɔ mɪlɛl ɔlɔ kɛlɛlɛmɛl mɑ mɔ mɔ ɛtʃɛmɗɛnɑm ɛlɛkidɑ`

**Final fine-tuned prediction:** `lokendu bedena mədi eiko alpa hamdik bulum dihamahem hamdolom dihamahem do asaipito amahaŋo ecub laɡido həmmo maledho loket ho hemhəm ŋɔ`

**Edit counts:** B0 S/D/I = 92/1/26; final F1 S/D/I = 36/16/26. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 has fewer edits than B0, while substitutions, deletions or insertions remain.

### Example 14 — 2.66 s; B0 90.62%, final F1 84.38%

**Dataset reference:** `alɟiba pura akma skul e kam alma`

**B0 zero-shot prediction:** `alʒgib`

**Final fine-tuned prediction:** `alɡe ba`

**Edit counts:** B0 S/D/I = 1/27/1; final F1 S/D/I = 2/25/0. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 deletions are its largest edit type; it omits part of the reference.

### Example 15 — 9.18 s; B0 84.85%, final F1 80.30%

**Dataset reference:** `komlaŋ ŋo si əki dena xeti to ŋul si tikma mana istatiŋ eicemab ie`

**B0 zero-shot prediction:** `kwamlɑŋ ɛmɑlɯʃiɛ ɑkiɾɑnɑ hɛtɪtɑ sɑm mʊlʃi ɑhɛɑvɑ tiʃ mɑmɑ nɑ ɛstɑ tɛnʋɑ tʃɑmɑ viɦɛ`

**Final fine-tuned prediction:** `komɛmləŋe ŋaŋoŋ siː akiː rana hetɨ t̪usa sam ŋul sija əjab tik məmana satik məma mana satin məcema bi`

**Edit counts:** B0 S/D/I = 38/1/17; final F1 S/D/I = 12/3/38. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 insertions exceed its other edit types combined; extra output remains a visible failure mode.

### Example 16 — 15.27 s; B0 99.24%, final F1 83.21%

**Dataset reference:** `mule pɛstɪsʌɪde kom juːz mado topom kadub pat̪a d̪amub mado mula kapopəm mado hoɡe muluɡe moha mara lekiŋəm moha bura eniŋəm mahate`

**B0 zero-shot prediction:** `mulɛpɛstisɛ ɛzɛpɛfɛ ɛkɛmɛjɔdʒmɔ mɛmɛdɔhɔl ɛtɔpɛmɛkɑdɔ patɑdɑ mɛmɑdɛ mɛnɛ tɑpɔ kɔmɛdɔ ɔgɛ hɔnɛgɑ maɦɑ mɑɦɑ lɛlɛkɛnɛ buɦɑ brɑlɛlɛ lɛlɛ ɪnjɛ mlɛnjɪ kɔ maɦɑɛtɛdɔ hɔ tɔpɑdɛd kɔulɛkw`

**Final fine-tuned prediction:** `ŋule pestisaːʈisaːɪzəb u akəm ɨuɟ məmaːd̪o ŋul sɪto pəmkado pata d̪a mamad̪ab məmaːd̪ab ŋule t̪apo kəmado hoɡe məleɡe məha mara lakiŋəm buhə bura leleleb aɲiŋɔm leɪniŋəm likɔ mahadu ho topo d̪ad̪adkuɬlik`

**Edit counts:** B0 S/D/I = 81/2/47; final F1 S/D/I = 29/4/76. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 insertions exceed its other edit types combined; extra output remains a visible failure mode.

### Example 17 — 4.70 s; B0 59.26%, final F1 166.67%

**Dataset reference:** `sat sati xosa e taŋari taŋe`

**B0 zero-shot prediction:** `sa sati osaiia ɗuani alita iɾi`

**Final fine-tuned prediction:** `saːtʰ saːtʰ saːtʰiː ho saː iː ɪ aːnd̪ kaːn naːliː t̪aːɲɔrjʊt̪aːjʈ`

**Edit counts:** B0 S/D/I = 7/3/6; final F1 S/D/I = 7/0/38. B0 EOS: yes; final F1 EOS: yes.

**Reading:** F1 makes more edits than B0 on this clip. This counterexample prevents treating the aggregate gain as universal.

### Example 18 — 16.03 s; B0 518.67%, final F1 130.67%

**Dataset reference:** `oririminko alenikunde anesujine anepajilo sominkokuŋe nermin kuŋe alenikune`

**B0 zero-shot prediction:** `ɔmɔl ɔfɑti ɗɑiɛ ɔmɔn ɑlbɛ mʄɛlɛjɑ ɔsɑ sɑmɔl pɹɑ ɛklɛl ɦɑpɛtɔ ɔ kɛɔn ɔ tɑbiɛn mɔ ɛvɛl ɔmɔl tsɔkɛɛl mɔl ɦɑ kɑm kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl kɛl`

**Final fine-tuned prediction:** `oɡatid̪a iː əmulab malaje hoso so so pula eklael hab bato oke oke ŋun ŋun tʰɔu mulsike ŋul mul nulu kam kiɯba mat hoɟujen pili`

**Edit counts:** B0 S/D/I = 52/0/337; final F1 S/D/I = 47/0/51. B0 EOS: no; final F1 EOS: yes.

**Reading:** B0 reached the generation cap without EOS; F1 ended with EOS and made fewer edits, though it remains inaccurate.

## How to use these examples

Use only a small readable subset in the main presentation; keep the full 18-pair gallery as a report appendix or repository companion. Quote the overall 683-clip CER from the Phase 8 report for the principal result. The examples show error patterns, not a frequency estimate, a linguistic translation, or evidence for post-test tuning.
