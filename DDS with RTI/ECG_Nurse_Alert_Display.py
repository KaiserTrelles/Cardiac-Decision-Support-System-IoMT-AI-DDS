#!/bin/env python

# ECG_Nurse_Alert_Display.py
# NURSE STATION DASHBOARD
# UPDATED: Uses 'colorama' to fix color issues on Windows Command Prompt.

import time
import sys
import rti.connextdds as dds

# --- ADDED: Colorama for Windows Colors ---
try:
    from colorama import init, Fore, Style
    # autoreset=True means we don't need to manually reset color after every print
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("Warning: 'colorama' not found. Colors will be disabled.")
    print("Run 'pip install colorama' to see colors.")

XML_FILENAME = "ECG_Monitoring_SystemApp_ECG_Suscriber.xml"

class NurseApp(dds.NoOpDomainParticipantListener):
    def __init__(self, xml_filename, participant_name):
        super().__init__()
        self._qos_provider = dds.QosProvider(xml_filename)
        self._participant = self._qos_provider.create_participant_from_config(participant_name)
        
        # Dynamic lookup for the Reader defined in XML
        self._reader = dds.DynamicData.DataReader(
            self._participant.find_datareader("Nurse_Subscriber::Alert_Reader")
        )
        
        self._reader.set_listener(NurseListener(), dds.StatusMask.DATA_AVAILABLE)

    def run(self):
        print("----------------------------------------------------")
        print("   NURSE STATION DASHBOARD - WAITING FOR ALERTS")
        print("----------------------------------------------------")
        while True:
            time.sleep(1)

class NurseListener(dds.DynamicData.NoOpDataReaderListener):
    def on_data_available(self, reader):
        try:
            samples = reader.take()
            for sample in samples:
                if sample.info.valid:
                    data = sample.data
                    sensor_id = data["sensor_id"]
                    diagnosis = data["diagnosis"]
                    level = data["alert_level"]
                    time_str = data["timestamp"]

                    # --- COLOR LOGIC ---
                    color_code = ""
                    reset_code = ""
                    
                    if COLORS_AVAILABLE:
                        if level == "EMERGENCY":
                            color_code = Fore.RED + Style.BRIGHT
                        elif level == "URGENT":
                            color_code = Fore.YELLOW + Style.BRIGHT
                        elif level == "CHECK_CAREFULLY":
                            color_code = Fore.CYAN + Style.BRIGHT
                        else:
                            # Normal / At Risk -> No special color (White/Grey)
                            color_code = Fore.WHITE
                        
                        reset_code = Style.RESET_ALL

                    # --- PRINT OUTPUT ---
                    if level == "EMERGENCY":
                        # Emergency gets "!!!" at the end
                        print(f"[{color_code}{level}{reset_code}] @ {time_str} | Patient: {sensor_id} | Dx: {diagnosis} !!!")
                    
                    elif level == "CHECK_CAREFULLY":
                        # Low confidence note
                        print(f"[{color_code}{level}{reset_code}] @ {time_str} | Patient: {sensor_id} | Dx: {diagnosis} (Low Confidence)")
                    
                    else:
                        # Standard Formatting for Normal/Urgent/At_Risk
                        # We pad the level string to align the columns nicely
                        padding = " " * (15 - len(level)) 
                        print(f"[{color_code}{level}{reset_code}]{padding} @ {time_str} | Patient: {sensor_id} | Dx: {diagnosis}")

        except Exception as e:
            print(f"Error reading alert: {e}")

def main():
    try:
        app = NurseApp(XML_FILENAME, "DomainParticipantLibrary::NurseParticipant")
        app.run()
    except KeyboardInterrupt:
        print("Nurse Station closed.")

if __name__ == '__main__':
    main()
