import os
import numpy as np
import scipy
import matplotlib.pylab as plt 
import ipywidgets as widgets
from IPython.display import display, clear_output


class load_signal():
    
    def __init__(self, path_to_signal):
        
        self.name = os.path.basename(path_to_signal)

        for file_name in os.listdir(path_to_signal):
            
            if file_name.startswith("mix_snr_"):
                suffix = file_name[len("mix_snr_") :-4]
                self.snr = float(suffix)
                self.mix = self.__read_signal__(path_to_signal + '/' + file_name)
            
            if file_name == "voice.wav":
                self.voice = self.__read_signal__(path_to_signal + '/' + file_name)
            
            if file_name == "noise.wav":
                self.noise = self.__read_signal__(path_to_signal + '/' + file_name)

    
    def __read_signal__(self, path_to_signal):
        
        fe, audio = scipy.io.wavfile.read(path_to_signal)
        
        # On nomralise les données afin d'éviter la saturation ou la distorsion du audio
        audio = audio/np.max(np.abs(audio)) 

        # On divise par l'écart type afin d'avoir un audio de variance = 1
        audio/=np.std(audio)
        
        return {
            "fe" : fe,
            "audio" : audio
            }
        

def interactive_spectrogramme(df):
    # Trier les SNR par ordre croissant
    sorted_snr = sorted(df["SNR"].unique())

    # Créer un menu déroulant pour les SNR uniques, trié
    snr_dropdown = widgets.Dropdown(
        options=sorted_snr,
        value=sorted_snr[0],  # Sélectionner le premier élément par défaut
        description="SNR:",
    )

    # Créer un menu déroulant pour les noms (sera mis à jour dynamiquement)
    name_dropdown = widgets.Dropdown(
        options=[],  # Initialement vide
        value=None,
        description="Nom:",
    )

    # Mettre à jour les options du menu déroulant des noms lorsque le SNR change
    def update_names(*args):
        selected_snr = snr_dropdown.value
        filtered_names = sorted(df[df["SNR"] == selected_snr]["Nom"].tolist())  # Trier les noms
        name_dropdown.options = filtered_names
        if filtered_names:  # Pré-sélectionner le premier nom si disponible
            name_dropdown.value = filtered_names[0]

    # Afficher le spectrogramme immédiatement lorsque le SNR ou le Nom change
    def on_value_change(change):
        clear_output(wait=True)  # Effacer la sortie précédente
        display(snr_dropdown, name_dropdown)  # Réafficher les widgets
        selected_snr = snr_dropdown.value
        selected_name = name_dropdown.value

        # Récupérer le signal correspondant
        path_to_signal = df[(df["SNR"] == selected_snr) & (df["Nom"] == selected_name)]["Path"].values[0]
        plot_spectrogramme(path_to_signal)

    # Lier la mise à jour des noms au changement du SNR
    snr_dropdown.observe(update_names, names="value")
    name_dropdown.observe(on_value_change, names="value")  # Observer les changements dans le nom

    # Initialiser les options de noms
    update_names()

    # Afficher les widgets
    display(snr_dropdown, name_dropdown)
    
    
def plot_spectrogramme(path_to_signal):

    signal = load_signal(path_to_signal)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))  
    
    f, t, Zxx = scipy.signal.stft(signal.voice["audio"], signal.voice["fe"], nperseg=signal.voice["fe"]*0.1)
    Zxx_log = 20*np.log10(np.abs(Zxx))
    pcm = axes[0,0].pcolormesh(t, f, Zxx_log, shading='gouraud')
    axes[0,0].set_ylabel('Fréquence [Hz]')
    axes[0,0].set_xlabel('Temps [s]')
    axes[0,0].set_title('Voice')
    fig.colorbar(pcm, ax=axes[0,0], orientation='vertical')
    
    axes[1,0].hist(Zxx_log.flatten(), 100)
    axes[1,0].set_xlabel('Valeurs prise par le spectrogramme en dB')
    axes[1,0].set_ylabel('Occurences')
    
    
    f, t, Zxx = scipy.signal.stft(signal.noise["audio"], signal.noise["fe"], nperseg=signal.noise["fe"]*0.1)
    Zxx_log = 20*np.log10(np.abs(Zxx))
    pcm = axes[0,1].pcolormesh(t, f, Zxx_log, shading='gouraud')
    axes[0,1].set_ylabel('Fréquence [Hz]')
    axes[0,1].set_xlabel('Temps [s]')
    axes[0,1].set_title('noise')
    fig.colorbar(pcm, ax=axes[0,1], orientation='vertical')
    
    axes[1,1].hist(Zxx_log.flatten(), 100)
    axes[1,1].set_xlabel('Valeurs prise par le spectrogramme en dB')
    axes[1,1].set_ylabel('Occurences')
    
    
    f, t, Zxx = scipy.signal.stft(signal.mix["audio"], signal.mix["fe"], nperseg=signal.mix["fe"]*0.1)
    Zxx_log = 20*np.log10(np.abs(Zxx))
    pcm = axes[0,2].pcolormesh(t, f,Zxx_log, shading='gouraud')
    axes[0,2].set_ylabel('Fréquence [Hz]')
    axes[0,2].set_xlabel('Temps [s]')
    axes[0,2].set_title(f'Mix - SNR = {signal.snr}')
    fig.colorbar(pcm, ax=axes[0,2], orientation='vertical')
    
    axes[1,2].hist(Zxx_log.flatten(), 100)
    axes[1,2].set_xlabel('Valeurs prise par le spectrogramme en dB')
    axes[1,2].set_ylabel('Occurences')
    
    
    plt.suptitle(f"Signal name = {signal.name}")
    plt.tight_layout()
    plt.show


def interactive_amplitude(df):
    # Trier les SNR par ordre croissant
    sorted_snr = sorted(df["SNR"].unique())

    # Créer un menu déroulant pour les SNR uniques, trié
    snr_dropdown = widgets.Dropdown(
        options=sorted_snr,
        value=sorted_snr[0],  # Sélectionner le premier élément par défaut
        description="SNR:",
    )

    # Créer un menu déroulant pour les noms (sera mis à jour dynamiquement)
    name_dropdown = widgets.Dropdown(
        options=[],  # Initialement vide
        value=None,
        description="Nom:",
    )

    # Mettre à jour les options du menu déroulant des noms lorsque le SNR change
    def update_names(*args):
        selected_snr = snr_dropdown.value
        filtered_names = sorted(df[df["SNR"] == selected_snr]["Nom"].tolist())  # Trier les noms
        name_dropdown.options = filtered_names
        if filtered_names:  # Pré-sélectionner le premier nom si disponible
            name_dropdown.value = filtered_names[0]

    # Afficher le spectrogramme immédiatement lorsque le SNR ou le Nom change
    def on_value_change(change):
        clear_output(wait=True)  # Effacer la sortie précédente
        display(snr_dropdown, name_dropdown)  # Réafficher les widgets
        selected_snr = snr_dropdown.value
        selected_name = name_dropdown.value

        # Récupérer le signal correspondant
        path_to_signal = df[(df["SNR"] == selected_snr) & (df["Nom"] == selected_name)]["Path"].values[0]
        plot_amplitude(path_to_signal)

    # Lier la mise à jour des noms au changement du SNR
    snr_dropdown.observe(update_names, names="value")
    name_dropdown.observe(on_value_change, names="value")  # Observer les changements dans le nom

    # Initialiser les options de noms
    update_names()

    # Afficher les widgets
    display(snr_dropdown, name_dropdown)

def plot_amplitude(path_to_signal):

    signal = load_signal(path_to_signal)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9))

    time = np.linspace(0, len(signal.mix['audio']) / signal.mix['fe'], num=len(signal.mix['audio']))

    axes[0].plot(time, signal.mix['audio'])
    axes[0].set_title("Mix, SNR={}".format(signal.snr))
    axes[0].set_ylabel("amplitude [dB]")
    axes[0].set_xlabel("time [s]")
    axes[0].grid(True)

    axes[1].plot(time, signal.voice['audio'])
    axes[1].set_title("Voice")
    axes[1].set_ylabel("amplitude [dB]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True)

    axes[2].plot(time, signal.noise['audio'])
    axes[2].set_title("Noise")
    axes[2].set_ylabel("amplitude [dB]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True)

    plt.tight_layout()

    plt.show()