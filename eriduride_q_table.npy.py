def load_q_table():
    global Q_TABLE
    if Q_TABLE is None:
        if os.path.exists(Q_TABLE_PATH):
            Q_TABLE = np.load(Q_TABLE_PATH)
            print("Q-table loaded successfully.")
        else:
            # FIX: Create an empty table if the file is missing
            # Taxi-v3 has 500 states and 6 actions
            st.warning(f"⚠️ {Q_TABLE_PATH} not found. Initializing with an untrained model.")
            Q_TABLE = np.zeros((500, 6)) 
    return Q_TABLE
