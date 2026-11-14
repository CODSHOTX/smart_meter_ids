Development of a Cyberattack Detection System for Smart Electricity Metering Networks
Statement of the Problem: Smart metering improves energy management but introduces cyber risks involving meter tampering, abnormal network traffic and unauthorized data manipulation.
Aim of the Study: To develop and evaluate an anomaly-based cyberattack detection system for smart electricity metering networks.
Research Objectives
1 Develop an anomaly detection model for identifying abnormal smart-meter communication or consumption patterns.
2 Implement the developed model using a public smart-grid/smart-meter dataset or a simulated metering environment.
3 Evaluate the implemented system using detection accuracy, precision, recall, F1-score, false-alarm rate and detection latency.
Recommended Algorithm / Technical Method: Isolation Forest for anomaly detection; Random Forest where labelled attack data are available.
Recommended Tools / Dataset / Test Environment: Python, Scikit-learn, Pandas; public smart-meter or smart-grid intrusion datasets/synthetic meter readings.
Performance Evaluation Metrics: Precision, recall, F1-score, false-alarm rate, where applicable and detection latency.
Expected Student Deliverable: A prototype that flags abnormal smart-meter behaviour and displays suspicious events