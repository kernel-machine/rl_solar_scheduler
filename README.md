# PREVISIONI UTILI SONO CON BATTERIA GRANDE
Durante i nostri esperimenti, abbiamo osservato una relazione critica tra la capacità della batteria del dispositivo e l'efficacia delle previsioni meteorologiche (es. previsioni LSTM).

- **Batteria Piccola (es. 10Ah):** Quando la batteria è molto piccola, riesce a contenere solo l'energia sufficiente per superare una singola notte. In questo scenario di "collo di bottiglia energetico", conoscere le previsioni del tempo per l'indomani non fornisce alcun vantaggio tattico. L'agente non può immagazzinare l'energia solare in eccesso oggi per sopravvivere a una tempesta domani, poiché la batteria raggiunge rapidamente il 100% e non può accumularne altra. Pertanto, un modello di baseline "cieco" (che usa solo l'ora e il livello della batteria) ottiene risultati ottimali limitandosi a processare in modo ingordo durante il giorno. L'aggiunta delle previsioni per l'LSTM si traduce solo in rumore, che rallenta l'addestramento e porta a prestazioni leggermente peggiori.
- **Batteria Grande (es. 50Ah+):** Quando la capacità della batteria viene aumentata, il sistema guadagna un "buffer energetico" sufficiente per pianificare su più giorni. In questo scenario, il modello LSTM può utilizzare efficacemente le previsioni del tempo per ridurre il processing prima di una tempesta di più giorni, accumulando una grande riserva di energia. Un modello di baseline, essendo cieco sul futuro, sprecherà questa energia nel processing immediato e finirà per scaricarsi e morire durante la tempesta. I nostri test dimostrano che con una batteria da 50Ah, il modello LSTM supera nettamente la baseline processando centinaia di migliaia di immagini in più nel corso del periodo di test.

Run senza previsioni
```bash
python -u main.py --gpu --use_solar --use_hour_minute --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 10 --test_year 2025 --n_env 10 --layer_width 64 --layer_depth 3 --lr_decay lin --train_days 350 --autostart --start_thr 0 --gamma 0.999 --device_idle_energy_w 2 --device_full_energy_w 8 --reward_shape 5 --buffer_weight 0.3 --battery_weight 0.7 --battery_ah 50
```

Run con previsioni
```bash
python -u main.py --gpu --use_solar --use_hour_minute --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 10 --test_year 2025 --n_env 10 --layer_width 64 --layer_depth 3 --lr_decay lin --train_days 350 --autostart --start_thr 0 --gamma 0.999 --device_idle_energy_w 2 --device_full_energy_w 8 --reward_shape 5 --buffer_weight 0.3 --battery_weight 0.7 --battery_ah 50 --lstm_prediction --lstm_model ghi_predictor.pth_lstm.pth --probabilistic_forecast
```

Invece se riduco la batteria a 10Ah le previsioni non servono.
Nota che ho usato reward shape 5, prima era 1.
Quindi se la batteria è troppo piccola, non ho abbastanza energia per pianificare uno scheduler efficace anche se conosco le previsioni.

### LSTM vs Real Forecast (Oracolo)
Durante gli esperimenti con batteria da 50Ah è emerso un risultato controintuitivo fondamentale:
* **Baseline**: ~2.82M immagini processate
* **Real Forecast (Oracolo Perfetto)**: ~2.82M immagini processate
* **LSTM (Previsione probabilistica)**: ~3.04M immagini processate

L'oracolo perfetto (real forecast) fallisce e ottiene gli stessi risultati della baseline per un motivo legato al campionamento. L'oracolo estrae il valore solare **istantaneo e puntuale** (es. esattamente alle 10:00:00). Se in quel preciso secondo passa una nuvola passeggera, il valore crolla a zero, inviando un segnale estremamente rumoroso e "spigoloso" all'agente RL (PPO).
Al contrario, il modello **LSTM** è stato pre-addestrato per predire il **GHI medio orario**, fungendo da potente estrattore di feature che *aggrega e pulisce* il segnale. Fornendo inoltre una previsione probabilistica pessimistica (es. quantile P10), offre all'agente RL una curva morbida e altamente affidabile dell'energia totale disponibile per quell'ora. Questo dimostra l'assoluta necessità di disaccoppiare l'estrazione delle feature meteorologiche (tramite LSTM) dallo scheduling vero e proprio (PPO).

## Dettagli Implementativi LSTM e Quantili

L'architettura del predittore LSTM è stata progettata per generare previsioni probabilistiche anziché semplici stime puntuali. A livello matematico, la rete non prevede un singolo valore per ogni ora futura, ma calcola una distribuzione di probabilità sfruttando la **Quantile Loss Function** in `train_predictor.py`.

### La Funzione di Perdita (Quantile Loss)
Durante il pre-addestramento supervisionato, l'LSTM è addestrato minimizzando contemporaneamente l'errore asimmetrico su tre quantili specifici:
* **P10 (10th percentile)**: Stima pessimistica (worst-case). C'è il 90% di probabilità che la radiazione solare effettiva sia *superiore* a questo valore.
* **P50 (50th percentile - Mediana)**: Stima centrale. Rappresenta l'andamento medio atteso.
* **P90 (90th percentile)**: Stima ottimistica (best-case). C'è solo il 10% di probabilità che la radiazione superi questo valore.

Il modello emette quindi un array di dimensione `(horizon * 3)`. Ad esempio, per un orizzonte di 24 ore, la rete produce 72 valori.

### Integrazione con l'Agente RL (PPO)
Il dettaglio implementativo fondamentale risiede in come questi dati vengono passati all'agente PPO all'interno di `environment.py`. 
Quando il flag `--probabilistic_forecast` è attivo, l'ambiente non passa tutti e tre i quantili, ma **estrae ed utilizza esclusivamente il quantile P10** (scartando P50 e P90):

```python
if self.probabilistic_forecast:
    # Estraiamo solo il quantile P10 (il primo elemento di ogni tripletta) per una previsione sicura/pessimistica
    predicted = predicted.reshape(self.nn_horizon, 3)[:, 0]
```

Passare unicamente la stima sicura (P10) all'agente RL fornisce un vantaggio tattico decisivo allo scheduler:
1. **Sicurezza Energetica:** L'agente RL programma i suoi job intensivi basandosi sul "worst-case scenario". Questo previene comportamenti rischiosi in cui l'agente svuota la batteria sperando in un sole che ha solo il 50% di probabilità di presentarsi.
2. **Robustezza all'Incertezza:** Il P10 filtra naturalmente l'incertezza (es. giornate parzialmente nuvolose). Se la varianza attesa è alta, il P10 crolla vicino a zero, forzando autonomamente l'agente a conservare l'energia.
3. **Mantenimento della Bassa Dimensionalità:** L'agente RL riceve in input solamente 24 feature temporali estremamente dense di informazione, anziché 72, garantendo una convergenza rapida e stabile della rete MLP di PPO.

## Setup the system :gear:

- Put [solcast2024.csv](https://github.com/user-attachments/files/21213151/solcast2024.csv) and [solcast2025.csv](https://github.com/user-attachments/files/21213140/solcast2025.csv) on the project folder (not in SRC) but in the project folder
- install the python libraries from the `requirements.txt`

## How to replicate the experiments :eyes:

Go to the `src` directory and run the `main.py`, the script accepts different parameters.

To reproduce the results of the paper you can run the software with the following parameters

For the PPO alghoritms:

```bash
python main.py --gpu --use_solar --use_hour_minute --use_images --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 64 --lr_decay lin --train_days 30 --autostart --start_thr 0.05 --use_pressure --use_humidity
```

For the A2C alghoritms:

```bash
python -u main.py --gpu --use_solar --use_hour_minute --use_images --steps 2000000 --update_steps 30 --alg a2c --lr 0.0007 --term_days 1 --test_year 2025 --n_env 10 --layer_width 64 --lr_decay lin --train_days 30 --autostart --start_thr 0.05 --use_pressure --use_humidity
```

It is possible to include or not include the parameters `--use_pressure`,`--use_humidity` to tests all the different scenarios reported in the paper.

For each run is created a folder in `runs`, containing all the plots of the executed experiments.

## Important files :file_folder:
- **main.py** contains the training and evaluation code
- **ilp_solver.py** contains the code to find the optimal solutions using the ILP
- **lib/environment.py** contains the RL Environment

## CSV filter

If you want to create a smaller CSV with only the columns used by `src/lib/solar/solar.py`, run:

```bash
python src/filter_solar_csv.py solcast2025.csv
```

This generates `solcast2025_solar_only.csv` in the same folder as the input file. You can also pass an explicit output path as the second argument.
