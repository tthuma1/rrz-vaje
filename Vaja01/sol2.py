import numpy as np
import matplotlib.pyplot as plt

# proj_fn = lambda f, p: (-f*p[0]/p[2], -f*p[1]/p[2])

### a)

# računam v cm
f = 10
Z = 1400
X = 500

x = f * X/Z

print(f"Višina drevesa na zadnji strani škatle: {x:.3f}")
print()

### b)

Z0 = 1000
Y = 250
f = 10
acc = 50
meritev_na_sekundo = 10
st_meritev = 30 * meritev_na_sekundo + 1

y = []
for i in range(st_meritev):
    t = i / meritev_na_sekundo
    Z = Z0 + 1/2 * acc * t**2 if t != 0 else Z0
    y.append(f * Y/Z)

print(y[:10], "...", y[-10:])

times = np.linspace(0, 30, st_meritev)

plt.clf()
plt.plot(times, y, ".")
plt.show()

### c)

# Zakaj se kamere z luknjico uporabljajo bolj kot teoretičen model in ne tudi v praksi?
#   Kot teoretični model se uporablja zaradi linearnosti - enostavno je računati padanje svetlobe na senzor, ker so samo ravne črte; nimamo leč, ki bi ukrivile svetlobo. V praksi se ne uporabljajo, ker so zaradi majhne luknjice potrebni dolgi časi osvetljevanja (če luknjico povečaš, dobiš zamegljeno sliko).

# Naštejte prednosti in slabosti kamer z lečami.
#
# Prednosti:
# - Manjši čas osvetlitve, ker z lečo zberejo več svetlobe in imajo večjo luknjo.
# - Lahko spreminjamo goriščno razdaljo in s tem kateri predmet je v fokusu.
# Slabosti:
# - Računanje padanja svetlobe ni linearno.
# - Slika se lahko popači, pride do npr. ukrivljanja ravnih črt na robovih slike.


### d)

# računam v mm
f = 60
Z = 95000

px_count = 200
DPI = 2500

# * 10 je za pretvorbo iz cm v mm
x = px_count / DPI * 2.54 * 10

X = x / f * Z
X_v_metrih = X / 1000

print()
print(f"Višina valja: {X_v_metrih:.3f}")

### d)

print()

# f = x * Z / X
fs = []

# Realne višina objekta: 15.4 cm
X = 15.4

# 36 cm -> 459 px
fs.append(459 * 36 / X)

# 47 cm -> 366 px
fs.append(366 * 47 / X)

# 55.5 cm -> 312 px
fs.append(312 * 55.5 / X)

# 64 cm -> 276 px
fs.append(276 * 64 / X)

# 75 cm -> 240 px
fs.append(240 * 75 / X)

# 86.5 cm -> 210 px
fs.append(210 * 86.5 / X)

f_avg = np.mean(fs)
print("Ocenjene goriščne razdalje [px]:", fs)
print("Povprečna goriščna razdalja [px]:", f_avg)
print("Povprečna napaka [px]:", np.array([abs(f_avg - f) for f in fs]))
print("Povprečna napaka [%]:", np.array([abs(f_avg - f) / f_avg * 100 for f in fs]))

# ---------------- testi -------------

# 58.5 cm -> 303 px
print()
ocena = f_avg * X / 58.5
real = 303
print("Ocena 58.5 cm ->", round(ocena, 2), "px")
print("Realno 58.5 cm ->", real, "px")
print("Napaka:", round(abs(real - ocena), 2), "px =", round(abs(real - ocena) / real * 100, 1), "%")

# 115 cm -> 162 px
print()
ocena = f_avg * X / 115
real = 162
print("Ocena 115 cm ->", round(ocena, 2), "px")
print("Realno 115 cm ->", real, "px")
print("Napaka:", round(abs(real - ocena), 2), "px =", round(abs(real - ocena) / real * 100, 1), "%")

