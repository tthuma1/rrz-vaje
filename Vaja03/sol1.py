### a)

# Kakšen je razpon vrednosti za vsakega od sklepov. V kakšni poziciji je roka, če so vsi sklepi na poziciji 0?

# SO101Leader:
# -------------------------------------------
# NAME            |    MIN |    POS |    MAX
# shoulder_pan    |    829 |   1998 |   3211
# shoulder_lift   |    809 |    815 |   3188
# elbow_flex      |    909 |   3068 |   3105
# wrist_flex      |    810 |   2691 |   3156
# wrist_roll      |     97 |   2061 |   3895
# gripper         |   2035 |   2047 |   3229

# SO101Follower:
# -------------------------------------------
# NAME            |    MIN |    POS |    MAX
# shoulder_pan    |    713 |   1970 |   3446
# shoulder_lift   |    827 |    937 |   3221
# elbow_flex      |    882 |   3101 |   3101
# wrist_flex      |    807 |   2767 |   3174
# wrist_roll      |    137 |   2094 |   3970
# gripper         |   2019 |   2057 |   3496

#    V kodi so te vrednosti v radianih od ničelne pozicije.

#    Pozicija 0 je, ko je roka obrnjena proti sredini, kaže gor in naprej (pod 90 stopinj; rama kaže gor, komolec kaže naprej,
#    zapestje kaže naprej), konica pa je stisnjena.
#
#    =----
#        |
#        |
#     ___|___

### b)

# Koliko natančno lahko kontrolirate pozicijo roke? Razmislite, kakšne so omejitve sklepov robota, ki ga
# uporabljate. Lahko dosežete poljubno točko v prostoru? Lahko s konico robota narišete poljubno obliko,
# ali vas konstrukcija roke omejuje?
#     Lahko dosežemo vse točke znotraj četrt krogle. S konico lahko rišemo poljubne oblike, dokler smo znotraj tega četrta krogle.
#     Nekatere točke na tobu izgledajo dostopne, ampak niso zares uporabne za roko.
#     Lahko narišemo polkog, navpično linijo. Težje naredimo vodoravno linijo.

# Kateri sklep bi rabili dodati, da lahko naredimo vodoravno linijo?
#     Rabili bi en rotacijski sklep pri zapestju, ki se vrti okoli navpične osi.