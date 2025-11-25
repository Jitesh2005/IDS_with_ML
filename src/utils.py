# src/utils.py
import pandas as pd
import numpy as np

KDD_COLUMNS = [
 "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
 "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
 "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
 "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
 "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
 "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]

def load_nslkdd(path):
    df = pd.read_csv(path, header=None)
    
    # if 43 columns -> last one is difficulty score
    if df.shape[1] == 43:
        df = df.drop(columns=[42])   # drop difficulty_score
    
    df.columns = KDD_COLUMNS
    df['label'] = df['label'].astype(str).str.strip()
    return df


def map_label_to_category(label):
    """Map detailed labels to the 5 high-level classes or binary."""
    dos = {"back","land","neptune","pod","smurf","teardrop","apache2","udpstorm","processtable","worm"}
    probe = {"satan","ipsweep","nmap","portsweep","mscan","saint"}
    r2l = {"ftp_write","guess_passwd","imap","phf","multihop","warezclient","warezmaster","sendmail","named","snmpgetattack","snmpguess","xlock","xsnoop","worm","imap","httptunnel"}
    u2r = {"buffer_overflow","loadmodule","perl","rootkit","sqlattack","xterm","ps"}
    lab = label.lower()
    if lab == 'normal':
        return 'normal'
    if lab in dos:
        return 'dos'
    if lab in probe:
        return 'probe'
    if lab in r2l:
        return 'r2l'
    if lab in u2r:
        return 'u2r'
    # unknowns -> attack
    return 'attack'
