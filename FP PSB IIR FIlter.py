import numpy as np  
import streamlit as st  
import matplotlib.pyplot as plt
import sys  

# Variable untuk sinyal 
st.title("Filter IIR ECG untuk Mendeteksi Denyut Jantung")  
data = st.file_uploader("Masukkan data sinyal", type=["txt"])
st.header("Parameter")
fs = st.number_input("Frekuensi Sampling", min_value=1.0, value=100.0, step=1.0)  
min_index = st.number_input("Range data min", min_value=0, value=0, step=1)  
max_index = st.number_input("Range data maks", min_value=1, value=2000, step=1)

# Variable untuk frekuensi cut-off
f_p = st.number_input("Fc Prefilter", min_value=0.1, value=100.0, step=0.5)  

# Variable untuk size window dan threshold
size = st.number_input("Ukuran Jendela", min_value=0.0000, value=0.05, step=0.0001)    
threshold = st.number_input("Threshold R-Peak", min_value=0.0000, value=0.4, step=0.0001)

# Variable untuk rentang data segmentasi
col1, col2, col3 = st.columns(3)
tstart_p = col1.number_input("Start time of P wave (ms)", min_value=0, value=19, step=1)
tstop_p = col1.number_input("End time of P wave (ms)", min_value=0, value=35, step=1)

tstart_qrs = col2.number_input("Start time of QRS complex (ms)", min_value=0, value=34, step=1)
tstop_qrs = col2.number_input("End time of QRS complex (ms)", min_value=0, value=46, step=1)

tstart_t = col3.number_input("Start time of T wave (ms)", min_value=0, value=45, step=1)
tstop_t = col3.number_input("End time of T wave (ms)", min_value=0, value=78, step=1)

if data is not None:
    try:
        raw = np.loadtxt(data, skiprows=1, usecols=1)
        ecg_signal = raw[int(min_index):int(max_index)]  
        st.subheader("Plotting Data Asli")  
        st.line_chart(ecg_signal)

        # Low pass filter IIR Orde 2
        def lpf(sig, fc_L, fs):  
            sig = np.copy(sig)  
            N = len(sig)  
            T = 1 / fs  
            Wc = 2 * np.pi * fc_L
            denominator_L = (4 / T**2) + (2 * np.sqrt(2) * Wc / T) + Wc**2
            a0_L = Wc**2 / denominator_L  
            a1_L = 2 * Wc**2 / denominator_L 
            a2_L = a0_L   
            b1_L = ((8 / T**2) - (2 * Wc**2)) / denominator_L 
            b2_L = ((4 / T**2) - (2 * np.sqrt(2) * Wc / T) + Wc**2) / denominator_L 
            y = np.zeros(N)  
            for n in range(2, N):  
                y[n] = (b1_L * y[n-1]) - (b2_L * y[n-2]) + (a0_L * sig[n]) + (a1_L * sig[n-1]) + (a2_L * sig[n-2])  
            return y  

        # High pass filter IIR Orde 2
        def hpf(sig, fc_H, fs):  
            sig = np.copy(sig)  
            N = len(sig)  
            T = 1 / fs  
            Wc = 2 * np.pi * fc_H 
            denominator_H = (4 / T**2) + (2 * np.sqrt(2) * Wc / T) + Wc**2  
            a0_H = (4 / T**2) / denominator_H  
            a1_H = (-8 / T**2) / denominator_H  
            a2_H = a0_H 
            b1_H = ((8 / T**2) - (2 * Wc**2)) / denominator_H  
            b2_H = ((4 / T**2) - (2 * np.sqrt(2) * Wc / T) + Wc**2) / denominator_H  
            y = np.zeros(N)  
            for n in range(2, N):  
                y[n] = (b1_H * y[n-1]) - (b2_H * y[n-2]) + (a0_H * sig[n]) + (a1_H * sig[n-1]) + (a2_H * sig[n-2])  
            return y  
    
        # plotting untuk prefilter
        sig_prefilter = lpf(ecg_signal, f_p, fs)
        st.subheader(f"Sinyal Setelah Diprefilter (fc = {f_p:.1f} Hz)")  
        st.line_chart(sig_prefilter)

        # Segmentasi P, QRS, dan T Wave
        def segmented_ecg(sig):
            p_wave = sig[tstart_p:tstop_p]
            index_p = np.arange(tstart_p, tstop_p)

            qrs_wave = sig[tstart_qrs:tstop_qrs]
            index_qrs = np.arange(tstart_qrs, tstop_qrs)

            t_wave = sig[tstart_t:tstop_t]
            index_t = np.arange(tstart_t, tstop_t)
            
            return index_t, index_p, index_qrs, p_wave, qrs_wave, t_wave

        # segmented ecg
        index_t, index_p, index_qrs, p_wave, qrs_wave, t_wave = segmented_ecg(sig_prefilter)

        # Plotting Segmented ECG
        st.header("Segmentasi P, QRS, dan T Wave")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(index_p, p_wave, color='red', label='Gelombang P')
        ax.plot(index_qrs, qrs_wave, color='green', label='Kompleks QRS')
        ax.plot(index_t, t_wave, color='blue', label='Gelombang T')
        ax.set_title("Segmentasi Gelombang P, QRS, dan T")
        ax.set_xlabel("Indeks Data")
        ax.set_ylabel("Amplitudo")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Fungsi DFT
        def dft(sig, fs):
            N = len(sig)
            Re = np.zeros(N) 
            Im = np.zeros(N)
            Mag = np.zeros(N)

            for k in range(N):
                for n in range(N):
                    omega = 2 * np.pi * k * n / N
                    Re[k] += sig[n] * np.cos(omega)
                    Im[k] -= sig[n] * np.sin(omega)

                Mag[k] = np.sqrt(Re[k] ** 2 + Im[k] ** 2)
            f = np.arange(0, N // 2) * fs / N
            return f, Mag[:N//2]

        #DFT untuk setiap segmen
        f_p, Mag_p = dft(p_wave, fs)
        f_qrs, Mag_qrs = dft(qrs_wave, fs)
        f_t, Mag_t = dft(t_wave, fs)
        
        # Plot hasil DFT
        st.header("DFT dari P, QRS, dan T Wave")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(f_p, Mag_p, color='red', label='Gelombang P')
        ax.plot(f_qrs, Mag_qrs, color='green', label='Kompleks QRS')
        ax.plot(f_t, Mag_t, color='blue', label='Gelombang T')
        ax.set_title("Spektrum Frekuensi dari P, QRS, dan T Wave")
        ax.set_xlabel("Frekuensi (Hz)")
        ax.set_ylabel("Magnitudo")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        #Frekuensi respon untuk QRS waveform  
        def peak_magnitude(f_qrs, Mag_qrs):  
            index_max = np.argmax(Mag_qrs)  
            fc_low = f_qrs[index_max]  

            if fc_low < 0.1:  
                fc_low = 0.1  
            
            mag_qrs_copy = np.copy(Mag_qrs)  
            mag_qrs_copy[index_max] = -np.inf  

            fc_high = f_qrs[np.argmax(mag_qrs_copy)]  
            return fc_low, fc_high  

        # Hitung frekuensi cutoff berdasarkan DFT QRS  
        fc_low, fc_high = peak_magnitude(f_qrs, Mag_qrs)
          
        st.subheader("Dapat diatur manual frekuensi cutoff")
        f_low = st.number_input("Fc Low pass", min_value=0.1, value=fc_low, step=0.5)  
        f_high = st.number_input("Fc High pass", min_value=0.1, value=fc_high, step=0.5)  

        # Frekuensi respon HPF
        def frequency_response_hpf(fs, fh):  
            T = 1 / fs  
            wc_hpf = 2 * np.pi * fh  
            num_points = 1000  
            omegas = np.linspace(0, np.pi, num_points)  
            frequencies = omegas * fs / (2 * np.pi)  
            magnitude_response_hpf = np.zeros(num_points)

            for i, omega in enumerate(omegas):
                numR_hpf = (4 / T**2) * (1 - 2 * np.cos(omega) + np.cos(2 * omega))
                numI_hpf = (4 / T**2) * (2 * np.sin(omega) - np.sin(2 * omega))
                denumR_hpf = (
                    wc_hpf**2 * (1 + 2 * np.cos(omega) + np.cos(2 * omega))
                    + np.sqrt(2) * wc_hpf * (2 / T) * (1 - np.cos(2 * omega))
                    + (4 / T**2) * (1 - 2 * np.cos(omega) + np.cos(2 * omega))
                )
                denumI_hpf = (
                    wc_hpf**2 * (2 * np.sin(omega) - np.sin(2 * omega))
                    + np.sqrt(2) * wc_hpf * (2 / T) * (1 - np.cos(2 * omega))
                    + (4 / T**2) * (2 * np.sin(omega) - np.sin(2 * omega))
                )
                hpf_complex_response = (numR_hpf + 1j * numI_hpf) / (denumR_hpf + 1j * denumI_hpf)
                magnitude_response_hpf[i] = np.abs(hpf_complex_response)

            return frequencies, magnitude_response_hpf

        # Frekuensi respon LPF
        def frequency_response_lpf(fs, fl):  
            T = 1 / fs  
            wc_lpf = 2 * np.pi * fl  
            num_points = 1000  
            omegas = np.linspace(0, np.pi, num_points)  
            frequencies = omegas * fs / (2 * np.pi)  
            magnitude_response_lpf = np.zeros(num_points)

            for i, omega in enumerate(omegas):
                numR_lpf = wc_lpf**2 * (1 + 2 * np.cos(omega) + np.cos(2 * omega))
                numI_lpf = -wc_lpf**2 * (2 * np.sin(omega) + np.sin(2 * omega))
                denumR_lpf = (
                    (4 / T**2) + (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2
                    - ((8 / T**2) - 2 * wc_lpf**2) * np.cos(omega)
                    + ((4 / T**2) - (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2) * np.cos(2 * omega)
                )
                denumI_lpf = (
                    ((8 / T**2) - 2 * wc_lpf**2) * np.sin(omega)
                    - ((4 / T**2) - (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2) * np.sin(2 * omega)
                )
                lpf_complex_response = (numR_lpf + 1j * numI_lpf) / (denumR_lpf + 1j * denumI_lpf)
                magnitude_response_lpf[i] = np.abs(lpf_complex_response)

            return frequencies, magnitude_response_lpf

        # Hitung frekuensi respon untuk HPF dan LPF
        frequencies_hpf, magnitude_response_hpf = frequency_response_hpf(fs, f_high)
        frequencies_lpf, magnitude_response_lpf = frequency_response_lpf(fs, f_low)
        
        # Plot respons frekuensi HPF dan LPF
        st.header("Respon Frekuensi HPF dan LPF pada QRS")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frequencies_hpf, magnitude_response_hpf, color='red', label='HPF')
        ax.plot(frequencies_lpf, magnitude_response_lpf, color='blue', label='LPF')
        ax.set_title("Respon Frekuensi pada QRS")
        ax.set_xlabel("Frekuensi (Hz)")
        ax.set_ylabel("Magnitudo")
        ax.set_ylim(0, 1.1) 
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Hitung dan plot respons frekuensi BPF
        magnitude_response_bpf = magnitude_response_hpf * magnitude_response_lpf
        frequencies_bpf = frequencies_hpf
        
        # Plot respons frekuensi BPF
        st.header("Respon Frekuensi BPF pada QRS")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frequencies_bpf, magnitude_response_bpf, color='green', label='BPF')
        ax.set_title("Respon Frekuensi pada QRS")
        ax.set_xlabel("Frekuensi (Hz)")
        ax.set_ylabel("Magnitudo")
        ax.set_ylim(0, np.max(magnitude_response_bpf) * 1.1) 
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Bandpass Filter
        def bpf(sig, f_low, f_high, fs):   
            low_pass = lpf(sig, f_low, fs)  
            band_pass = hpf(low_pass, f_high, fs)  
            return band_pass  
        
        # plotting low pass filter output
        sig_lpf = lpf(sig_prefilter, f_low, fs)
        st.subheader(f"Sinyal Low-Pass (fc = {f_low:.1f} Hz)")  
        st.line_chart(sig_lpf)
        
        # plotting high pass filter output
        sig_hpf = hpf(sig_prefilter, f_high, fs) 
        st.subheader(f"Sinyal High-Pass (fc = {f_high:.1f} Hz)")  
        st.line_chart(sig_hpf) 

        # plotting bandpass filter output
        sig_bpf = bpf(sig_prefilter, f_low, f_high, fs)
        st.subheader(f"Sinyal Band-Pass ({f_low:.1f}–{f_high:.1f} Hz)")
        st.line_chart(sig_bpf)

        # Derivative 
        def derivative_filter(x, fs):
            b = np.array([-1, -2, 0, 2, 1]) / 8
            return np.convolve(x, b, mode='same')
        
        deriv = derivative_filter(sig_bpf, fs)  
        st.subheader("Output Signal Derivatif")
        st.line_chart(deriv)

        # Square
        def apply_square(signal):
            return np.square(signal)

        squared_signal = apply_square(deriv)  
        st.subheader("Output SIgnal squared")
        st.line_chart(squared_signal)

        # Moving Window Integration
        def moving_window_integration(signal, fs):
            result = np.copy(signal)
            win_size = round(size * fs)  
            sum_val = np.sum(signal[:win_size])  

            for j in range(win_size):
                result[j] = sum_val / win_size
        
            for index in range(win_size, len(signal)):  
                sum_val += signal[index] / win_size
                sum_val -= signal[index - win_size] / win_size
                result[index] = sum_val
            return result

        mwi_signal = moving_window_integration(squared_signal, fs)
        st.subheader("Output Signal Windowing")
        st.line_chart(mwi_signal)

        # Cari amplitudo maksimum dari sinyal hasil windowing 
        max_amplitude = np.max(mwi_signal)
        threshold = 0.65 * max_amplitude
        st.write(f"Amplitude max sinyal: {max_amplitude}")
        st.write(f"Threshold: {threshold}")
        
        # Deteksi R-Peak  
        def find_r_peaks(signal):  
            peaks = []  
            for i in range(1, len(signal) - 1):  
                if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:  
                    peaks.append(i)  
            return peaks  

        r_peaks = find_r_peaks(mwi_signal)  
        st.write(f"Jumlah R-Peaks yang Terdeteksi: {len(r_peaks)}")

        # Menghitung bpm 
        if len(r_peaks) > 1:  
            rr_intervals = np.diff(r_peaks) / fs  
            heart_rate = 60 / np.mean(rr_intervals)  
        else:  
            heart_rate = 0  

        st.write(f"Bpm: {heart_rate:.2f} BPM")  

        # Plotting Lokasi R Peak  
        plt.figure(figsize=(20, 8), dpi=100)  
        plt.plot(mwi_signal, color='blue') 
        plt.axhline(threshold, color='gray', ls='--', label='Ambang')       
        plt.scatter(r_peaks, mwi_signal[r_peaks], color='red', s=100, marker='*')  
        plt.xlabel('domain waktu')  
        plt.ylabel('amplitudo')  
        plt.title("Lokasi R Peak")  
        st.pyplot(plt)  
        
    except Exception as e:
        st.error(f"Masukan kembali data: {e}")