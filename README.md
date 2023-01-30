# Event-based Three-factor Local Plasticity (ETLP)

> Neuromorphic perception with event-based sensors, asynchronous hardware and spiking neurons is showing promising results for real-time and energy-efficient inference in embedded systems. The next promise of brain-inspired computing is to enable adaptation to changes at the edge with online learning. However, the parallel and distributed architectures of neuromorphic hardware based on co-localized compute and memory imposes locality constraints to the on-chip learning rules. We propose in this work the Event-based Three-factor Local Plasticity (ETLP) rule that uses (1) the pre-synaptic spike trace, (2) the post-synaptic membrane voltage and (3) a third factor in the form of projected labels with no error calculation, that also serve as update triggers. We apply ETLP with feedforward and recurrent spiking neural networks on visual and auditory event-based pattern recognition, and compare it to Back-Propagation Through Time (BPTT) and eProp. We show a competitive performance in accuracy with a clear advantage in the computational complexity for ETLP. We also show that when using local plasticity, threshold adaptation in spiking neurons and a recurrent topology are necessary to learn spatio-temporal patterns with a rich temporal structure. Finally, we provide a proof of concept hardware implementation of ETLP on FPGA to highlight the simplicity of its computational primitives and how they can be mapped into neuromorphic hardware for online learning with low-energy consumption and real-time interaction.

## Citation

If you find ETLP useful in your work, please cite the following [source](https://arxiv.org/abs/2301.08281):

```
@misc{Quintana_etal,
  doi = {10.48550/ARXIV.2301.08281},  
  url = {https://arxiv.org/abs/2301.08281},
  author = {Quintana, Fernando M. and Perez-Peña, Fernando and Galindo, Pedro L. and Neftci, Emre O. and Chicca, Elisabetta and Khacef, Lyes},
  keywords = {Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ETLP: Event-based Three-factor Local Plasticity for online learning with neuromorphic hardware},
  publisher = {arXiv},
  year = {2023},
}
```
