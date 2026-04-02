import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
#  f(x) = 2x - 3cos(x) + 1
# ============================================================
def f(x):
    return 2*x - 3*np.cos(x) + 1

def df(x):
    return 2 + 3*np.sin(x)

def ddf(x):
    return 3*np.cos(x)

a, b, eps = 0, 1, 0.0001

# ============================================================
# 1. ORALIQNI TENG IKKIGA BO'LISH USULI (Bisection)
# ============================================================
def bisection(a, b, eps):
    steps = []
    k = 0
    while True:
        c = (a + b) / 2
        steps.append((k, a, b, c, f(c)))
        if abs(b - a) < eps:
            break
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        k += 1
    return c, k, steps

# ============================================================
# 2. VATARLAR USULI (Secant / Chord method)
# ============================================================
def vatarlar(a, b, eps):
    steps = []
    k = 0
    while True:
        c = a - f(a) * (b - a) / (f(b) - f(a))
        steps.append((k, a, b, c, f(c)))
        if abs(f(c)) < eps:
            break
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        k += 1
        if k > 1000:
            break
    return c, k, steps

# ============================================================
# 3. URINMALAR USULI (Newton's method / Tangent)
# ============================================================
def urinmalar(a, b, eps):
    # Boshlang'ich nuqta: f(x0)*f''(x0) > 0 bo'lgan tomoni
    x = b if f(b) * ddf(b) > 0 else a
    steps = []
    k = 0
    while True:
        x_new = x - f(x) / df(x)
        steps.append((k, x, f(x), x_new))
        if abs(x_new - x) < eps:
            x = x_new
            break
        x = x_new
        k += 1
        if k > 1000:
            break
    return x, k, steps

# ============================================================
# HISOBLASH
# ============================================================
root1, iter1, steps1 = bisection(a, b, eps)
root2, iter2, steps2 = vatarlar(a, b, eps)
root3, iter3, steps3 = urinmalar(a, b, eps)

# ============================================================
# NATIJALARNI CHIQARISH
# ============================================================
print("=" * 60)
print("  f(x) = 2x - 3cos(x) + 1,  oraliq: [0, 1],  ε = 0.0001")
print("=" * 60)

print("\n📌 1. ORALIQNI TENG IKKIGA BO'LISH USULI")
print(f"   Ildiz:        x ≈ {root1:.6f}")
print(f"   Iteratsiya:   {iter1} ta")
print(f"   Tekshirish:   f({root1:.6f}) = {f(root1):.2e}")
print(f"\n   Qadamlar jadvali (dastlabki 5 ta):")
print(f"   {'k':>3} | {'a':>10} | {'b':>10} | {'c=(a+b)/2':>12} | {'f(c)':>12}")
print("   " + "-"*55)
for row in steps1[:5]:
    print(f"   {row[0]:>3} | {row[1]:>10.6f} | {row[2]:>10.6f} | {row[3]:>12.6f} | {row[4]:>12.2e}")

print("\n📌 2. VATARLAR USULI (Chord/Secant method)")
print(f"   Ildiz:        x ≈ {root2:.6f}")
print(f"   Iteratsiya:   {iter2} ta")
print(f"   Tekshirish:   f({root2:.6f}) = {f(root2):.2e}")
print(f"\n   Qadamlar jadvali (dastlabki 5 ta):")
print(f"   {'k':>3} | {'a':>10} | {'b':>10} | {'c':>12} | {'f(c)':>12}")
print("   " + "-"*55)
for row in steps2[:5]:
    print(f"   {row[0]:>3} | {row[1]:>10.6f} | {row[2]:>10.6f} | {row[3]:>12.6f} | {row[4]:>12.2e}")

print("\n📌 3. URINMALAR USULI (Newton's method)")
print(f"   Ildiz:        x ≈ {root3:.6f}")
print(f"   Iteratsiya:   {iter3} ta")
print(f"   Tekshirish:   f({root3:.6f}) = {f(root3):.2e}")
print(f"\n   Qadamlar jadvali:")
print(f"   {'k':>3} | {'x_k':>12} | {'f(x_k)':>14} | {'x_(k+1)':>12}")
print("   " + "-"*50)
for row in steps3:
    print(f"   {row[0]:>3} | {row[1]:>12.6f} | {row[2]:>14.2e} | {row[3]:>12.6f}")

print("\n" + "=" * 60)
print("  YAKUNIY TAQQOSLASH")
print("=" * 60)
print(f"  {'Usul':<30} | {'Ildiz':^12} | {'Iteratsiya':^10}")
print("  " + "-"*58)
print(f"  {'Teng ikkiga bo\'lish':<30} | {root1:^12.6f} | {iter1:^10}")
print(f"  {'Vatarlar usuli':<30} | {root2:^12.6f} | {iter2:^10}")
print(f"  {'Urinmalar (Newton)':<30} | {root3:^12.6f} | {iter3:^10}")
print("=" * 60)

# ============================================================
# GRAFIK
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("f(x) = 2x − 3cos(x) + 1  |  Ildiz: x ≈ {:.6f}".format(root1), fontsize=13, fontweight='bold')

# --- Chap: funksiya grafigi ---
ax1 = axes[0]
x_vals = np.linspace(-0.5, 1.5, 400)
y_vals = f(x_vals)
ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = 2x − 3cos(x) + 1')
ax1.axhline(0, color='black', linewidth=0.8)
ax1.axvline(0, color='black', linewidth=0.8)
ax1.axvspan(0, 1, alpha=0.1, color='green', label='Oraliq [0, 1]')
ax1.plot(root1, 0, 'ro', markersize=10, zorder=5, label=f'Ildiz x ≈ {root1:.5f}')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('Funksiya grafigi va ildiz', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- O'ng: iteratsiyalar taqqoslash ---
ax2 = axes[1]
iters_b = [abs(row[3] - root1) for row in steps1]
iters_v = [abs(row[3] - root2) for row in steps2]
iters_n = [abs(row[3] - root3) for row in steps3]

ax2.semilogy(range(len(iters_b)), iters_b, 'b-o', markersize=5, label=f"Teng ikkiga bo'lish ({iter1} iter)")
ax2.semilogy(range(len(iters_v)), iters_v, 'g-s', markersize=5, label=f"Vatarlar usuli ({iter2} iter)")
ax2.semilogy(range(len(iters_n)), iters_n, 'r-^', markersize=5, label=f"Urinmalar / Newton ({iter3} iter)")
ax2.set_xlabel('Iteratsiya soni', fontsize=12)
ax2.set_ylabel('Xato (log shkala)', fontsize=12)
ax2.set_title("Usullar tezligini taqqoslash", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('natija_grafik.png', dpi=150, bbox_inches='tight')
print("\n✅ Grafik saqlandi: natija_grafik.png")