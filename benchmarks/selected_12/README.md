# Selected-12 Poster Benchmark

This directory is the cleaned 12-paper benchmark subset used for paper-writing and qualitative comparison.

- 6 HEP papers + 6 non-HEP papers.
- Each paper has two posters: `ours.png` from Paper Poster Harness and `p2p.png` from the corrected P2P baseline run.
- The numbering has been compacted to `01`–`12`; `manifest.csv` keeps the old benchmark labels and arXiv IDs.
- P2P posters for No. 01,02,03,04,05,06,08,10,11 were rerun with real figure caches instead of full-page PDF screenshot fallback. No. 07,09,12 kept their original P2P baseline because those inputs were already acceptable.

## Files

```text
by_paper/<NN_short>/ours.png
by_paper/<NN_short>/p2p.png
contact_sheets/selected_12_ours_contact.jpg
contact_sheets/selected_12_p2p_contact.jpg
contact_sheets/selected_12_p2p_vs_ours_contact.jpg
manifest.csv
manifest.json
```
