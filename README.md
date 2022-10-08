# mapsv-anpr

## Què és ANPR?

En essència, els sistemes de "Reconeixement automàtic de matrícules" (ANPR) s'utilitzen per detectar i reconèixer automàticament matrícules en imatges. A partir d'aquí, la matrícula identificada es pot utilitzar per buscar informació sobre el propietari de l'automòbil. Els sistemes ANPR s'utilitzen principalment per al control d'accés a parkings tot i que  l'ús principal d'ANPR és per multar als conductors per haver realitzar infraccions de tràfic.


És important comprendre que no hi ha una solució única per a tots els ANPR!.  Sempre que puguem aplicar coneixements a priori sobre el domini del problema a resoldre  (ja sigui visió per ordinador, aprenentatge automàtic o teixir una bufanda per a la nostra àvia), aquest coneixement ens ajudarà a resoldre el problema de manera més efectiva i precisa. Per aquest repte ens centrarem en matricules españoles. Des del darrer canvi de format en les matrícules aquestes tenen 4 xifres seguides de 3 lletres

Llavors, ara que sabem què és ANPR en realitat, repassem breument els 4 passos necessaris per a construir qualsevol sistema ANPR. Òbviament, revisarem cada un d'aquests passos amb més detall a la resta d'aquest mòdul 


## Objectius:

 Familiaritzeu-vos amb els termes Reconeixement automàtic de matrícules .

 Els quatre passos per construir qualsevol sistema ANPR:

  1. Adquisició d’una foto

  2. Localització

  3. Segmentació

  4. Reconeixement


## Funcionament:

 Per fer aquest projecte més dinàmic, hem implementat UI molt senzilla que permet:
  - Generar prediccions donat un fitxer o un conjunt d'ells
  - Visualitzar els resultats de cada pas de l'algoritme
  - Guardar les imatges resultants de cada pas i de la predicció final
  
  
## Environment
A partir del fitxer mapsv-anpr/requirements.txt pots replicar l'environment amb totes les dependencies que hem necessitat durant el projecte. Abans de poder executar el programa, has de:

  1. Crear un venv (recomanem fer-ho a la carpeta root del projecte): python -m venv venv
  2. Activar l'environment: source venc/bin/activate
  3. Instal·lar els paquets necessaris: python -m pip install -r requirements.txt
   
## Exemple:

 cd mapsv-anpr/src \
 python main.py
 
 A partir d'aqui, segueix les indicacions de la UI.
 
 Exemple: prediccions per totes les imatges de la nostra DB guardant els resultats de cada pas.\
 mode --> 'predict'\
 write --> 'y'\
 path --> '/home/gerard/PycharmProjects/mapsv-anpr/images/raw_images/' (full path to you project images)
 
 Podras veure els resultats de cada pas seguint la governancia de la carpeta 'images'
 
 Exemple: visualització interactiva de cada pas del nostre algoritme donada una imatge.\
 mode --> 'visualize'\
 write --> 'n'\
 path --> '../images/raw_images/1621HRH.jpg'\
 A partir d'aqui ves indicant 'y' o 'n' per visualitzar cadascun dels passos.
 
