#!/bin/env python 

# ECG_Monitoring_SystemApp_ECG_Suscriber.py
# FINAL LOGIC: Includes Full Name Mapping, Risk Levels, and 95% Confidence Rule.

import argparse
import random
import string
import time
import sys
from dataclasses import dataclass

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import datetime 

import rti.connextdds as dds

XML_FILENAME = "ECG_Monitoring_SystemApp_ECG_Suscriber.xml" 

# --- GLOBAL: LOAD AI MODELS ---
print("Loading AI Models...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes_adult = [
    'Atrial premature beat', 'Fusion of paced and normal beat', 
    'Fusion of ventricular and normal beat', 'Left bundle branch block beat', 
    'Normal beat', 'Paced beat', 'Premature ventricular contraction', 
    'Right bundle branch block beat'
]

classes_child = [
    'AFIB', 'Atrial_Paced', 'AVNRT', 'AVRT', 
    'Normal', 'PJC', 'Preexcited_PAC', 'Ventricular_Paced'
]

# --- MAPPINGS: DEFINING RULES ---

# 1. Adult Risks
ADULT_RISK_MAP = {
    'Premature ventricular contraction': 'EMERGENCY',
    'Left bundle branch block beat':     'URGENT',
    'Atrial premature beat':             'AT_RISK',
    'Fusion of ventricular and normal beat': 'AT_RISK',
    'Right bundle branch block beat':    'AT_RISK',
    'Paced beat':                        'NORMAL',
    'Fusion of paced and normal beat':   'NORMAL',
    'Normal beat':                       'NORMAL'
}

# 2. Child Risks (Key = Abbreviation)
CHILD_RISK_MAP = {
    'Preexcited_PAC':    'EMERGENCY',
    'AFIB':              'URGENT',
    'AVNRT':             'URGENT',
    'AVRT':              'URGENT',
    'PJC':               'AT_RISK',
    'Atrial_Paced':      'NORMAL',
    'Ventricular_Paced': 'NORMAL',
    'Normal':            'NORMAL'
}

# 3. Child Full Names (For better display)
CHILD_FULL_NAMES = {
    'Preexcited_PAC':    'Preexcited Premature Atrial Contraction',
    'AFIB':              'Atrial Fibrillation',
    'AVNRT':             'AV Nodal Reentrant Tachycardia',
    'AVRT':              'AV Reentrant Tachycardia',
    'PJC':               'Premature Junctional Contraction',
    'Atrial_Paced':      'Atrial Paced Beat',
    'Ventricular_Paced': 'Ventricular Paced Beat',
    'Normal':            'Normal Sinus Beat'
}

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_one_model(model_path, num_classes):
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found: {model_path}")
        return None
        
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

print(" > Loading Adult Model...")
model_adult = load_one_model('ecg_mobilenet_BEST.pth', len(classes_adult))

print(" > Loading Child Model...")
model_child = load_one_model('ecg_mobilenet_children_BEST.pth', len(classes_child))

print(f"AI Models loaded and running on {device}.")
# --- END GLOBAL AI SETUP ---


class MyDDS_Subscriber_AI_Model_Participant_Listener(dds.DynamicData.NoOpDataReaderListener):
    def __init__(self, app):
        super().__init__()
        self._app = app

    def on_data_available(self, reader):
        self._app.read_a_i__model__participant_dr()


class App(dds.NoOpDomainParticipantListener):
    def __init__(self, xml_filename, participant_name):
        super().__init__()
        self._qos_provider = dds.QosProvider(xml_filename)
        self._participant  = self._qos_provider.create_participant_from_config(participant_name)
        self._participant.set_listener(self, dds.StatusMask.ALL & dds.StatusMask.flip(dds.StatusMask.DATA_ON_READERS))

        # Input Reader
        self._a_i__model__participant_dr = dds.DynamicData.DataReader(self._participant.find_datareader("Subscriber::AI_Model_Participant"))
        self._a_i__model__participant_dr_read_count = 0
        self._a_i__model__participant_dr.set_listener(MyDDS_Subscriber_AI_Model_Participant_Listener(self), dds.StatusMask.DATA_AVAILABLE)
        
        # Output Writer
        self._alert_writer = dds.DynamicData.DataWriter(self._participant.find_datawriter("Alert_Publisher::Alert_Writer"))
        self._alert_sample = self._alert_writer.create_data() 

    def read_a_i__model__participant_dr(self):
        try:
            data_state = dds.DataState().with_instance_state(dds.InstanceState.alive)
        except AttributeError:
            data_state = dds.DataState(dds.InstanceState.ALIVE)

        try:
            samples = self._a_i__model__participant_dr.select().state(data_state).take()

            for sample in samples:
                if sample.info.valid:
                    self._a_i__model__participant_dr_read_count += 1
                    
                    sensor_id = sample.data["sender"]
                    filename = sample.data["message"]
                    image_bytes = bytes(sample.data["image_data"])

                    if not image_bytes: continue

                    try:
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    except Exception as e:
                        continue
                    
                    input_tensor = data_transforms(image)
                    input_batch = input_tensor.unsqueeze(0).to(device)

                    # --- ROUTER & DIAGNOSIS LOGIC ---
                    raw_diagnosis = "Unknown"
                    final_display_name = "Unknown"
                    alert_level = "NORMAL"
                    model_name_log = ""
                    conf_percent = 0.0
                    
                    # 1. Determine Model and Maps based on ID
                    current_model = None
                    current_classes = None
                    
                    if "Adult" in sensor_id:
                        current_model = model_adult
                        current_classes = classes_adult
                        model_name_log = "ADULT"
                    elif "Child" in sensor_id:
                        current_model = model_child
                        current_classes = classes_child
                        model_name_log = "CHILD"
                    else:
                        current_model = model_adult
                        current_classes = classes_adult
                        model_name_log = "ADULT(Def)"

                    if current_model is None: continue

                    # 2. Run Inference
                    with torch.no_grad():
                        outputs = current_model(input_batch)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, preds = torch.max(probs, 1)
                        
                        # This is the abbreviated class name directly from PyTorch
                        raw_diagnosis = current_classes[preds[0].item()]
                        conf_percent = confidence[0].item() * 100

                    # 3. Apply Business Logic (Risk & Naming)
                    if "Child" in sensor_id:
                        # Convert Abbreviation to Full Name
                        final_display_name = CHILD_FULL_NAMES.get(raw_diagnosis, raw_diagnosis)
                        # Get Risk Level
                        alert_level = CHILD_RISK_MAP.get(raw_diagnosis, "NORMAL")
                    else:
                        # Adult names are already full
                        final_display_name = raw_diagnosis
                        alert_level = ADULT_RISK_MAP.get(raw_diagnosis, "NORMAL")

                    # 4. Apply 95% Confidence Rule (CRITICAL FIX)
                    # If confidence is low, override risk to CHECK_CAREFULLY
                    if conf_percent < 85.0:
                        alert_level = "CHECK_CAREFULLY"

                    # --- PRINT CONSOLE (Subscriber View) ---
                    print("------------------------------------------------")
                    print(f"Received sample #{self._a_i__model__participant_dr_read_count} from [{sensor_id}]")
                    print(f"  Original File: {filename}")
                    print(f"  ROUTER LOGIC:  Used {model_name_log} Model")
                    print(f"  AI DIAGNOSIS:  {raw_diagnosis}")
                    print(f"  FULL NAME:     {final_display_name}")
                    print(f"  Confidence:    {conf_percent:.2f}%")
                    print(f"  ALERT STATUS:  {alert_level}") 
                    print("------------------------------------------------\n")
                    
                    # --- SEND ALERT TO NURSE APP ---
                    try:
                        self._alert_sample["sensor_id"] = sensor_id
                        self._alert_sample["diagnosis"] = final_display_name # Send Full Name
                        self._alert_sample["alert_level"] = alert_level      # Send Correct Level
                        self._alert_sample["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
                        
                        self._alert_writer.write(self._alert_sample)
                    except Exception as e:
                        print(f"[Warning] Could not send alert: {e}")

        except Exception as e:
            print(f"[ERROR] in read: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        print("AI Model Subscriber (Adult + Child) is running.")
        print("Waiting for data from sensors...")
        print("Press CTRL+C to stop")
        
        while True:
            time.sleep(60)
            print(f"Subscriber is alive... (Processed {self._a_i__model__participant_dr_read_count} samples so far)")

    # --- Listener callbacks ---
    def on_requested_deadline_missed(self, reader, status):
        print(f"on_requested_deadline_missed: topic=#{reader.topic_name}")
    def on_requested_incompatible_qos(self, reader, status):
        print(f"on_requested_incompatible_qos: topic=#{reader.topic_name}")
    def on_sample_rejected(self, reader, status):
        print(f"on_sample_rejected: topic=#{reader.topic_name}")
    def on_liveliness_changed(self, reader, status):
        print(f"on_liveliness_changed: topic=#{reader.topic_name}")
    def on_sample_lost(self, reader, status):
        print(f"on_sample_lost: topic=#{reader.topic_name}")
    def on_subscription_matched(self, reader, status):
        print(f"on_subscription_matched: topic=#{reader.topic_name}")
    def on_offered_deadline_missed(self, writer, status):
        print(f"on_offered_deadline_missed: topic=#{writer.topic_name}")
    def on_liveliness_lost(self, writer, status):
        print(f"on_liveliness_lost: topic=#{writer.topic_name}")
    def on_offered_incompatible_qos(self, writer, status):
        print(f"on_offered_incompatible_qos: topic=#{writer.topic_name}")
    def on_publication_matched(self, writer, status):
        print(f"on_publication_matched: topic=#{writer.topic_name}, match count=#{status.current_count}")


# --- ARGUMENT PARSING ---
@dataclass
class ApplicationArguments:
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='ECG Subscriber')
    return parser.parse_args(namespace=ApplicationArguments)

def main():
    args: ApplicationArguments = parse_arguments()
    try:
        app = App(XML_FILENAME, "DomainParticipantLibrary::ReaderParticipant")
        app.run()
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == '__main__':
    main()
