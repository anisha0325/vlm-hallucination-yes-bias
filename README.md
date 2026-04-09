# System-Mediated Attention Imbalances Make Vision-Language Models Say Yes
*Tsan Tsai Chan*, [*Varsha Suresh*](https://sites.google.com/view/varsha-suresh/), [*Anisha Saha*](https://anisha0325.github.io/), [*Michael Hahn*](https://www.mhahn.info/), [*Vera Demberg*](https://www.uni-saarland.de/lehrstuhl/demberg/members/verademberg.html)  
 

🚧 **Code is coming soon!** Stay tuned.  
📄 Paper: [arXiv 2601.12430](https://arxiv.org/pdf/2601.12430)

---

## 📌 Overview

<div style="text-align: justify"

VLM hallucinations may stem not just from weak image attention, but from excessive system-level attention that crowds out image and text signals. Rebalancing attention away from the system modality significantly reduces “yes-bias” and improves model reliability beyond image-focused fixes.

---

## 🧠 Abstract

<div style="text-align: justify"

Vision-language model (VLM) hallucination
is commonly linked to imbalanced allocation
of attention across input modalities: system,
image and text. However, existing mitigation
strategies tend towards an image-centric inter-
pretation of these imbalances, often prioritis-
ing increased image attention while giving less
consideration to the roles of the other modal-
ities. In this study, we evaluate a more holis-
tic, system-mediated account, which attributes
these imbalances to functionally redundant sys-
tem weights that reduce attention to image and
textual inputs. We show that this framework
offers a useful empirical perspective on the yes-
bias, a common form of hallucination in which
VLMs indiscriminately respond yes. Causally
redistributing attention from the system modal-
ity to image and textual inputs substantially
suppresses this bias, often outperforming exist-
ing approaches. We further present evidence
suggesting that system-mediated attention im-
balances contribute to the yes-bias by encour-
aging a default reliance on coarse input repre-
sentations, which are effective for some tasks
but ill-suited to others. Taken together, these
findings firmly establish system attention as a
key factor in VLM hallucination and highlight
its potential as a lever for mitigation


## 📬 Contact

For questions, feel free to contact:

*Tsan Tsai Chan* 
📧 [tsch00001@stud.uni-saarland.de](mailto:tsch00001@stud.uni-saarland.de)  
*Anisha Saha*
📧 [ansaha@mpi-inf.mpg.de](mailto:ansaha@mpi-inf.mpg.de)  
🔗 [anisha0325.github.io](https://anisha0325.github.io/)

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{saha2025mustreasonbenchmarkdiagnosingpragmatic,
      title={MUStReason: A Benchmark for Diagnosing Pragmatic Reasoning in Video-LMs for Multimodal Sarcasm Detection}, 
      author={Anisha Saha and Varsha Suresh and Timothy Hospedales and Vera Demberg},
      year={2025},
      eprint={2510.23727},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23727}, 
}
```
---

_This repository will be updated shortly. Thank you for your interest!_

