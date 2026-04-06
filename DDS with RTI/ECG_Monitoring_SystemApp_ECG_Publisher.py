#!/bin/env python 

# ECG_Monitoring_SystemApp_ECG_Publisher.py
# ADAPTED to send mixed Adult and Child data randomly.

import argparse
import random
import string
import time
import sys
import os  # Added for file paths
from dataclasses import dataclass

import rti.connextdds as dds

# --- CONFIGURATION ---
XML_FILENAME = "ECG_Monitoring_SystemApp_ECG_Publisher.xml"

# --- EDITED: Added both paths ---
PATH_ADULT = r"C:\Users\Usuario\Desktop\ECG_Training_First_Dataset\First_Dataset_Training\test"
PATH_CHILD = r"C:\Users\Usuario\Desktop\ECG_Training_Second_Dataset\data_ecg\test"

# --- HELPER FUNCTION ---
def read_image_as_bytes(filename):
    """Reads an image file and returns a list of integer bytes."""
    try:
        with open(filename, "rb") as f:
            return list(f.read()) 
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

# --- SETTER FUNCTION (Unchanged) ---
def ECG_Message_setter(sample, count, sensor_id, image_path):
    sample["sender"] = sensor_id
    if image_path:
        sample["message"] = os.path.basename(image_path)
        image_bytes = read_image_as_bytes(image_path)
        if image_bytes:
            sample["image_data"] = image_bytes
        else:
            sample["image_data"] = [] 
    else:
        sample["message"] = "No Image"
        sample["image_data"] = []
    sample["count"] = count


# App
# ----------------------------------------------------------------------------
class App(dds.NoOpDomainParticipantListener):
    def __init__(self, xml_filename, participant_name):
        super().__init__()
        self._qos_provider = dds.QosProvider(xml_filename)
        self._participant  = self._qos_provider.create_participant_from_config(participant_name)
        self._participant.set_listener(self, dds.StatusMask.ALL & dds.StatusMask.flip(dds.StatusMask.DATA_ON_READERS))

        # --- DataWriter(s) (Unchanged) ---
        self._e_c_g_sensor1__participant_dw = dds.DynamicData.DataWriter(self._participant.find_datawriter("Publisher::ECGSensor1_Participant"))
        self._e_c_g_sensor1__participant_dw_data = self._e_c_g_sensor1__participant_dw.create_data()

        self._e_c_g_sensor2__participant_dw = dds.DynamicData.DataWriter(self._participant.find_datawriter("Publisher::ECGSensor2_Participant"))
        self._e_c_g_sensor2__participant_dw_data = self._e_c_g_sensor2__participant_dw.create_data()

        self._e_c_g_sensor3__participant_dw = dds.DynamicData.DataWriter(self._participant.find_datawriter("Publisher::ECGSensor3_Participant"))
        self._e_c_g_sensor3__participant_dw_data = self._e_c_g_sensor3__participant_dw.create_data()

        self._e_c_g_sensor4__participant_dw = dds.DynamicData.DataWriter(self._participant.find_datawriter("Publisher::ECGSensor4_Participant"))
        self._e_c_g_sensor4__participant_dw_data = self._e_c_g_sensor4__participant_dw.create_data()

    # --- Write Functions (Unchanged) ---
    def write_e_c_g_sensor1__participant_dw(self, count, sensor_id, image_path):
        ECG_Message_setter(self._e_c_g_sensor1__participant_dw_data, count, sensor_id, image_path)
        self._e_c_g_sensor1__participant_dw.write(self._e_c_g_sensor1__participant_dw_data)

    def write_e_c_g_sensor2__participant_dw(self, count, sensor_id, image_path):
        ECG_Message_setter(self._e_c_g_sensor2__participant_dw_data, count, sensor_id, image_path)
        self._e_c_g_sensor2__participant_dw.write(self._e_c_g_sensor2__participant_dw_data)

    def write_e_c_g_sensor3__participant_dw(self, count, sensor_id, image_path):
        ECG_Message_setter(self._e_c_g_sensor3__participant_dw_data, count, sensor_id, image_path)
        self._e_c_g_sensor3__participant_dw.write(self._e_c_g_sensor3__participant_dw_data)

    def write_e_c_g_sensor4__participant_dw(self, count, sensor_id, image_path):
        ECG_Message_setter(self._e_c_g_sensor4__participant_dw_data, count, sensor_id, image_path)
        self._e_c_g_sensor4__participant_dw.write(self._e_c_g_sensor4__participant_dw_data)


    # --- EDITED Run Loop ---
    def run(self, period_microsec, max_iterations, max_keys):
        print("------------------------------------------------")
        print("   ECG SENSOR SIMULATION STARTED")
        print("   Modes: [ADULT] and [CHILDREN]")
        print("------------------------------------------------")
        print("Press CTRL+C to stop\n")
        
        count = 0
        
        # --- EDITED: Load Adult Images ---
        images_adult = []
        if os.path.exists(PATH_ADULT):
            for root, dirs, files in os.walk(PATH_ADULT):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images_adult.append(os.path.join(root, file))
        print(f"Loaded {len(images_adult)} Adult images.")

        # --- EDITED: Load Child Images ---
        images_child = []
        if os.path.exists(PATH_CHILD):
            for root, dirs, files in os.walk(PATH_CHILD):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images_child.append(os.path.join(root, file))
        print(f"Loaded {len(images_child)} Child images.")

        if not images_adult and not images_child:
            print("CRITICAL ERROR: No images found in either path.")
            return

        # 2. Main loop
        while ((max_iterations == -1) or (count < max_iterations)):
            print(f"\n--- Iteration #{count} ---")
            
            # --- EDITED: Logic to pick 4 sensors (Adult or Child) ---
            # We loop 4 times (for Sensor 1, 2, 3, 4)
            
            current_batch_data = [] # Store tuples of (sensor_id, filepath)

            for i in range(1, 5): # 1 to 4
                # Flip a coin: True = Adult, False = Child
                is_adult = random.choice([True, False])
                
                # Safety check: if one list is empty, force the other
                if not images_adult: is_adult = False
                if not images_child: is_adult = True

                if is_adult:
                    user_type = "Adult"
                    img_path = random.choice(images_adult)
                    # --- EDITED: Specific Sender ID for Subscriber to recognize ---
                    sender_id = f"Adult_Sensor_{i}" 
                else:
                    user_type = "Children"
                    img_path = random.choice(images_child)
                    # --- EDITED: Specific Sender ID for Subscriber to recognize ---
                    sender_id = f"Child_Sensor_{i}"

                # --- EDITED: Print the User Type as requested ---
                print(f"[{sender_id}] User: {user_type} | Writing {os.path.basename(img_path)}...")
                current_batch_data.append((sender_id, img_path))

            # Write data (unpacking our batch list)
            self.write_e_c_g_sensor1__participant_dw(count, current_batch_data[0][0], current_batch_data[0][1])
            self.write_e_c_g_sensor2__participant_dw(count, current_batch_data[1][0], current_batch_data[1][1])
            self.write_e_c_g_sensor3__participant_dw(count, current_batch_data[2][0], current_batch_data[2][1])
            self.write_e_c_g_sensor4__participant_dw(count, current_batch_data[3][0], current_batch_data[3][1])

            count += 1
            time.sleep(period_microsec/1000000)

    # --- All listener callbacks are kept as-is ---
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
        print(f"on_lliveliness_lost: topic=#{writer.topic_name}")
    def on_offered_incompatible_qos(self, writer, status):
        print(f"on_offered_incompatible_qos: topic=#{writer.topic_name}")
    def on_publication_matched(self, writer, status):
        print(f"on_publication_matched: topic=#{writer.topic_name}, match count=#{status.current_count}")


# --- ARGUMENT PARSING (Unchanged) ---
@dataclass
class ApplicationArguments:
    loop_period_msec = 1000
    max_iterations = -1
    max_key_count = 10

def parse_arguments():
    parser = argparse.ArgumentParser(description='ECG_Monitoring_SystemApp_ECG_Publisher Connext DDS Application')
    parser.add_argument('-l', '--loop_period_msec', type=int, help='Number of milliseconds to wait in main spin loop')
    parser.add_argument('-i', '--max_iterations', type=int, help='Number iterations to perform before terminating (-1=unlimited)')
    parser.add_argument('-k', '--max_key_count', type=int, help='Number keys to use when publishing data')
    return parser.parse_args(namespace=ApplicationArguments)

# --- MAIN ENTRY POINT (Unchanged) ---
def main():
    args: ApplicationArguments = parse_arguments()
    try:
        app = App(XML_FILENAME, "DomainParticipantLibrary::WriterParticipant")
        app.run(args.loop_period_msec*1000, args.max_iterations, args.max_key_count)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
