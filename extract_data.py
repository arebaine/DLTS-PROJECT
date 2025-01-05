import os
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
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

def plotlosses_history(df, model_name):

    plt.figure(figsize=(8, 5))
    plt.plot(df.index +1, df['train_loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(df.index+1, df['valid_loss'], label='Valid Loss', color='orange', marker='x')

    # Ajout de titres et de légendes
    plt.title('Train and Validation Loss for {}'.format(model_name))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Affichage
    plt.show()

def SISDR_by_SNR(model, test_data , affichage = True, model_name = "nom du model", device = 'cpu', spectrogram = True):

    device = torch.device(device)
    model.to(device)

    si_sdr_dict = { str(int(snr)) : [] for snr in test_data.SNR.unique()}

    si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
    for i in range(len(test_data)):
        snr = test_data.iloc[i]['SNR']
        path_to_data = test_data.iloc[i]['Path']

        _, voice_audio = scipy.io.wavfile.read(os.path.join(path_to_data, 'voice.wav'))
        _, mix_audio = scipy.io.wavfile.read(path_to_data+'/mix_snr_{:.0f}.wav'.format(snr))
        voice_audio, mix_audio = torch.Tensor(voice_audio), torch.Tensor(mix_audio)

        if spectrogram:
            Sxx_mix = torch.stft(mix_audio, n_fft=800, window=torch.hann_window(800), return_complex=True).unsqueeze(0).unsqueeze(0).to(device)
            Sxx_mix_ampl = torch.abs(Sxx_mix)
            Sxx_mix_ampl = Sxx_mix_ampl/torch.max(Sxx_mix_ampl)
            pred = model(Sxx_mix_ampl)

            Sxx_voice_reconstruct = pred*Sxx_mix/Sxx_mix_ampl
            audio_reconstruct = torch.istft(Sxx_voice_reconstruct[0,0,:,:].to('cpu'), n_fft = 800, window=torch.hann_window(800)).to(device)
        
        else :
            audio_reconstruct = model(mix_audio.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0).squeeze(0)
            
        si_sdr_dict[str(int(snr))].append(float(si_sdr(voice_audio.to(device),audio_reconstruct).cpu().detach().numpy()))

    if affichage :
        plot_SISDR_by_SNR(si_sdr_dict, model_name)
    
    return si_sdr_dict

        
def plot_SISDR_by_SNR(si_sdr_dict, model_name):

    sorted_keys = sorted(si_sdr_dict.keys())  # Liste des clés triées
    sorted_values = [si_sdr_dict[key] for key in sorted_keys]

    # Création du boxplot
    plt.figure(figsize=(8, 5))
    plt.boxplot(sorted_values, tick_labels=sorted_keys, patch_artist=True)

    # Ajout de titres et d'annotations
    plt.title('SI-SDR by SNR for {}'.format(model_name))
    plt.xlabel('SNR')
    plt.ylabel('SI-SDR')
    plt.grid(True)

    # Affichage
    plt.show() 

def interactive_reconstruction(df, model, spectrogram = True):
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
        plot_reconstruction(model, path_to_signal, spectrogram)

    # Lier la mise à jour des noms au changement du SNR
    snr_dropdown.observe(update_names, names="value")
    name_dropdown.observe(on_value_change, names="value")  # Observer les changements dans le nom

    # Initialiser les options de noms
    update_names()

    # Afficher les widgets
    display(snr_dropdown, name_dropdown)

def plot_reconstruction(model, path_to_signal, spectrogram = True, ):

    model.to("cpu")

    signal = load_signal(path_to_signal)

    Smix = torch.stft(torch.tensor(signal.mix["audio"]), n_fft = 800, window = torch.hann_window(800), return_complex = True).unsqueeze(0)
    Svoice = torch.stft(torch.tensor(signal.voice["audio"]), n_fft = 800, window = torch.hann_window(800), return_complex = True).unsqueeze(0)

    if spectrogram:
        Smix_ampl = torch.abs(Smix)
        Smix_ampl = Smix_ampl/torch.max(Smix_ampl)
        mask = model(Smix_ampl.unsqueeze(0))
        svoice_reconst = mask*Smix/Smix_ampl
        svoice_reconst = svoice_reconst/torch.max(torch.abs(svoice_reconst))

        audio = torch.istft(svoice_reconst[0,0,:,:].to('cpu'), n_fft = 800, window=torch.hann_window(800))
    else :
        mix_audio = torch.tensor(signal.mix["audio"]).unsqueeze(0).unsqueeze(0)
        audio = model(mix_audio).squeeze(0).squeeze(0)

        svoice_reconst = torch.stft(audio, n_fft = 800, window = torch.hann_window(800), return_complex = True).unsqueeze(0).unsqueeze(0)

    fig, axes = plt.subplots(2,3, figsize = (25,12))

    pcm0 = axes[0,0].pcolormesh(20*np.log10(np.abs(Smix[0,:,:].detach().numpy())), cmap='viridis')
    fig.colorbar(pcm0, ax = axes[0,0])
    axes[0,0].set_title("Spectrogramme du mix")

    pcm1 = axes[0,1].pcolormesh(20*np.log10(np.abs(Svoice[0,:,:].detach().numpy())), cmap='viridis')
    fig.colorbar(pcm1, ax = axes[0,1])
    axes[0,1].set_title("Spectrogramme du ground truth")

    pcm2 = axes[0,2].pcolormesh(20*np.log10(np.abs(svoice_reconst[0,0,:,:].detach().numpy())), cmap='viridis')
    fig.colorbar(pcm2, ax = axes[0,2])
    axes[0,2].set_title("Spectrogramme reconstruit")

    time = np.array(list(range(80000)))/8000
    axes[1,0].plot(time, signal.mix["audio"])
    axes[1,0].set_title("Mixte Signal")

    axes[1,1].plot(time, signal.voice["audio"])
    axes[1,1].set_title("Voice Signal")

    axes[1,2].plot(time, audio.detach().numpy())
    axes[1,2].set_title("Reconstructed Voice Signal")

    plt.tight_layout()
    plt.show()

def compute_clustering(clustering_model, dc_model, signal, device="cpu"):
    Smix = torch.stft(torch.tensor(signal), n_fft = 800, window = torch.hann_window(800), return_complex = True).unsqueeze(0).to(device)
    Smix_abs = torch.abs(Smix)
    # Compute the embeddings of the spectrograms
    dc_model.to(device)
    Smix_emb = dc_model(Smix_abs/torch.max(Smix_abs))
    _,T,F,D = Smix_emb.shape
    # Perform clustering
    model = clustering_model(n_clusters=2, random_state=42)
    model.fit(Smix_emb[0,:,:,:].view(-1,D).to("cpu").detach())
    
    class_assignments = torch.Tensor(model.predict(Smix_emb[0,:,:,:].view(-1,D).to("cpu").detach())).reshape(T,F)
    S1 = class_assignments*(Smix_abs[0,:,:].to("cpu"))
    S2 = (1-class_assignments)*(Smix_abs[0,:,:].to("cpu"))

    # Apply the two masks to the original spectrogram
    S1com = S1*(Smix[0,:,:].to("cpu")/Smix_abs[0,:,:].to("cpu"))
    S2com = S2*(Smix[0,:,:].to("cpu")/Smix_abs[0,:,:].to("cpu"))

    # Reconstruction of the audios
    au1 = torch.istft(S1com, n_fft = 800, window = torch.hann_window(800))
    au2 = torch.istft(S2com, n_fft = 800, window = torch.hann_window(800))

    return au1, au2, S1com, S2com

def plot_reconstruction_dc(model, path_to_signal, clustering_model):

    model.to("cpu")

    signal = load_signal(path_to_signal)
    Smix = torch.stft(torch.tensor(signal.mix["audio"]), n_fft = 800, window = torch.hann_window(800), return_complex = True).unsqueeze(0)
    Svoice = torch.stft(torch.tensor(signal.voice["audio"]), n_fft = 800, window = torch.hann_window(800), return_complex = True).unsqueeze(0)
    Snoise = torch.stft(torch.tensor(signal.noise["audio"]), n_fft = 800, window = torch.hann_window(800), return_complex = True).unsqueeze(0)

    au1, au2, S1com, S2com = compute_clustering(clustering_model, model, signal.mix["audio"], device="cpu")


    fig, axes = plt.subplots(4,2, figsize = (25,12))

    pcm0 = axes[0,0].pcolormesh(np.abs(Smix[0,:,:].detach().numpy()), cmap='viridis')
    fig.colorbar(pcm0, ax = axes[0,0])
    axes[0,0].set_title("Spectrogramme du mix")

    pcm1 = axes[0,1].pcolormesh(np.abs(Svoice[0,:,:].detach().numpy()), cmap='viridis')
    fig.colorbar(pcm1, ax = axes[0,1])
    axes[0,1].set_title("Spectrogramme du ground truth")

    pcm2 = axes[1,0].pcolormesh(np.abs(S1com.detach().numpy()), cmap='viridis')
    fig.colorbar(pcm2, ax = axes[1,0])
    axes[1,0].set_title("Spectrogramme du premier cluster")

    pcm2 = axes[1,1].pcolormesh(np.abs(S2com.detach().numpy()), cmap='viridis')
    fig.colorbar(pcm2, ax = axes[1,1])
    axes[1,1].set_title("Spectrogramme du deuxième cluster")

    time = np.array(list(range(80000)))/8000
    axes[2,0].plot(time, signal.mix["audio"])
    axes[2,0].set_title("Mixte Signal")

    axes[2,1].plot(time, signal.voice["audio"])
    axes[2,1].set_title("Voice Signal")

    axes[3,0].plot(time, au1.detach().numpy())
    axes[3,0].set_title("Signal 1er cluster")

    axes[3,1].plot(time, au2.detach().numpy())
    axes[3,1].set_title("Signal 2eme cluster")

    plt.tight_layout()
    plt.show()

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    print(f"Total parameters: {total_params}")

def SISDR_by_SNR_DC(model, clustering_model, test_data, affichage = True, model_name = "Deep Clustering", device = 'cpu', spectrogram = True):
    '''
    Compute the Si-SDR by SNR, only for the deep clustering model
    '''
    device = torch.device(device)
    model.to(device)

    si_sdr_dict = { str(int(snr)) : [] for snr in test_data.SNR.unique()}

    si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
    for i in range(len(test_data)):
        snr = test_data.iloc[i]['SNR']
        path_to_data = test_data.iloc[i]['Path']
        signal = load_signal(path_to_data)
        _, voice_audio = scipy.io.wavfile.read(os.path.join(path_to_data, 'voice.wav'))
        _, mix_audio = scipy.io.wavfile.read(path_to_data+'/mix_snr_{:.0f}.wav'.format(snr))
        voice_audio = torch.Tensor(voice_audio)
        au1,au2,S1,S2 = compute_clustering(clustering_model, model, signal.mix["audio"], device=device)
        si_sdr_dict[str(int(snr))].append(float(np.max([si_sdr(voice_audio.to(device),au1.to(device)).cpu().detach().numpy(),si_sdr(voice_audio.to(device),au2.to(device)).cpu().detach().numpy()])))
    
    if affichage :
        plot_SISDR_by_SNR(si_sdr_dict, model_name)
    
    return si_sdr_dict