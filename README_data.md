# Data Setup

## AMI Meeting Corpus

The AMI Meeting Corpus is **not included** in this repository.  
It is freely available for academic research from:

- **Public release:** http://corpus.amiproject.org/
- **Direct from authors:** contact the AMI project (see corpus website)

---

## Expected Folder Structure

After downloading, place the corpus so that `config.py` can find it.  
Set `AMI_ROOT` in `config.py` to the root of the corpus. The scripts  
expect the following layout inside that root:

```
ami_corpus/
│
├── words/
│   ├── ES2002a.A.words.xml
│   ├── ES2002a.B.words.xml
│   └── ...
│
├── dialogueActs/
│   ├── ES2002a.A.dialog-act.xml
│   ├── ES2002a.B.dialog-act.xml
│   └── ...
│
├── decision/
│   └── manual/
│       ├── ES2002a/
│       │   └── ES2002a.decision.xml
│       └── ...
│
├── meetings.xml
└── participants.xml
```

---

## Analysis Sample

This thesis uses the **120 scenario meetings** from the AMI corpus  
(meeting IDs beginning with ES, IS, or TS), filtered to those with  
complete word-level transcripts and dialogue-act annotations.

Meetings with missing or incomplete annotation layers are excluded,  
yielding the final analysis sample described in Section 3.1 of the thesis.

---

## Demographic Metadata

The fairness audit (RQ5) requires a `data/demographic_metadata.csv` file  
with gender composition per meeting. This file is **not included** because  
it is derived from the AMI `participants.xml` file, which contains  
personally identifiable information.

To recreate it, parse `participants.xml` and compute the proportion of  
female participants per meeting group. The expected format is:

```
meeting_id,pct_female
ES2002a,0.25
ES2002b,0.25
ES2002c,0.25
ES2002d,0.25
IS1000a,0.50
...
```

`pct_female` is the proportion of the four participants who are female  
(0.0, 0.25, 0.50, 0.75, or 1.0).

Meetings with `pct_female >= 0.5` are classified as **gender-balanced**;  
meetings with `pct_female < 0.5` are classified as **gender-imbalanced**.

---

## Target Variables

The five performance dimensions used as regression targets are derived  
from post-meeting Likert-scale questionnaires included in the AMI corpus.  
The composite scores are computed from the questionnaire items described  
in Section 3.2 of the thesis and stored in `outputs/y_performance_final.csv`.

The expected format of `y_performance_final.csv` is:

```
meeting_id,overall_performance,satisfaction,cohesiveness,leadership,information_processing
ES2002a,5.50,5.25,5.75,5.00,5.50
...
```
