"""
Analisis de Comportamientos Emergentes
"""
import os
import sys

if sys.platform == 'win32':
    os.system('')

from collections import defaultdict

import numpy as np
import torch
from zeta_attentive_predictive import ZetaAttentivePredictive

def analizar_emergencia():
    print("\n" + "=" * 70)
    print("   ANALISIS DE COMPORTAMIENTOS EMERGENTES")
    print("=" * 70)

    system = ZetaAttentivePredictive(n_cells=100)

    # Estructuras para analisis
    datos = {
        'consciousness': [],
        'surprise_L1': [],
        'surprise_L2': [],
        'surprise_L3': [],
        'coherence': [],
        'intensity': [],
        'error_attention': [],
        'memory_importance': [],
        'dominant_arch': [],
        'context_threat': [],
        'context_opportunity': [],
    }

    # Ejecutar 500 pasos con estimulos variados
    n_steps = 500

    for step in range(n_steps):
        # Patron mixto con cambios abruptos
        if step < 100:
            stimulus = torch.rand(4)
        elif step < 200:
            # Amenaza sostenida
            stimulus = torch.tensor([0.1, 0.7, 0.1, 0.1])
        elif step < 300:
            # Transicion a oportunidad
            t = (step - 200) / 100
            stimulus = torch.tensor([0.1 + 0.6*t, 0.7 - 0.6*t, 0.1, 0.1])
        elif step < 400:
            # Patron ciclico rapido
            phase = (step % 10) / 10 * 2 * np.pi
            stimulus = torch.tensor([
                0.5 + 0.3*np.sin(phase),
                0.5 + 0.3*np.cos(phase),
                0.3,
                0.3
            ], dtype=torch.float32)
        else:
            # Caos final
            stimulus = torch.rand(4) * 2

        result = system.step(stimulus)

        # Recolectar datos
        datos['consciousness'].append(result['consciousness'])
        datos['surprise_L1'].append(result['errors']['L1']['surprise'])
        datos['surprise_L2'].append(result['errors']['L2']['surprise'])
        datos['surprise_L3'].append(result['errors']['L3']['meta_surprise'])
        datos['coherence'].append(result['attention']['coherence'])
        datos['intensity'].append(result['attention']['intensity'])
        datos['error_attention'].append(result['attention']['error'].detach().numpy())
        datos['dominant_arch'].append(result['observation']['dominant'].value)
        datos['context_threat'].append(result['attention']['context']['threat'])
        datos['context_opportunity'].append(result['attention']['context']['opportunity'])

        # Importancia de memoria
        if len(system.attention.memory_buffer) > 0:
            avg_importance = np.mean(system.attention.memory_buffer.importance)
            datos['memory_importance'].append(avg_importance)
        else:
            datos['memory_importance'].append(0)

    # ===== ANALISIS DE EMERGENCIA =====

    print("\n" + "-" * 70)
    print("   1. ADAPTACION A SORPRESA")
    print("-" * 70)

    # Correlacion entre sorpresa y cambios en atencion
    surprise_total = np.array(datos['surprise_L1']) + np.array(datos['surprise_L2'])
    intensity = np.array(datos['intensity'])

    # Calcular cambios en intensidad despues de sorpresas altas
    high_surprise_idx = np.where(surprise_total > np.percentile(surprise_total, 75))[0]

    intensity_after_surprise = []
    for idx in high_surprise_idx:
        if idx + 5 < len(intensity):
            intensity_after_surprise.append(np.mean(intensity[idx:idx+5]))

    intensity_normal = np.mean(intensity)
    intensity_post_surprise = np.mean(intensity_after_surprise) if intensity_after_surprise else 0

    print(f"\n  Intensidad normal:           {intensity_normal:.4f}")
    print(f"  Intensidad post-sorpresa:    {intensity_post_surprise:.4f}")
    print(f"  Cambio:                      {intensity_post_surprise - intensity_normal:+.4f}")

    if intensity_post_surprise > intensity_normal:
        print("\n  >> EMERGENTE: La atencion se ENFOCA despues de sorpresas")
    else:
        print("\n  >> La atencion no muestra adaptacion clara a sorpresas")

    print("\n" + "-" * 70)
    print("   2. CONSOLIDACION DE MEMORIA")
    print("-" * 70)

    importance = np.array(datos['memory_importance'])

    # Buscar picos de importancia
    if len(importance) > 10:
        importance_diff = np.diff(importance)
        spikes = np.where(importance_diff > np.std(importance_diff) * 2)[0]

        print(f"\n  Importancia promedio:        {np.mean(importance):.4f}")
        print(f"  Importancia maxima:          {np.max(importance):.4f}")
        print(f"  Picos de consolidacion:      {len(spikes)}")

        if len(spikes) > 0:
            print(f"  Ubicacion de picos:          {spikes[:10]}...")
            print("\n  >> EMERGENTE: Memoria consolida eventos importantes")

    print("\n" + "-" * 70)
    print("   3. HOMEOSTASIS DE COHERENCIA")
    print("-" * 70)

    coherence = np.array(datos['coherence'])

    # Analizar estabilidad de coherencia
    coherence_var = np.var(coherence)
    coherence_mean = np.mean(coherence)

    # Buscar recuperacion despues de caidas
    low_coherence_idx = np.where(coherence < np.percentile(coherence, 25))[0]
    recovery_times = []

    for idx in low_coherence_idx:
        for future in range(1, min(20, len(coherence) - idx)):
            if coherence[idx + future] > coherence_mean:
                recovery_times.append(future)
                break

    print(f"\n  Coherencia promedio:         {coherence_mean:.4f}")
    print(f"  Varianza de coherencia:      {coherence_var:.6f}")
    print(f"  Caidas detectadas:           {len(low_coherence_idx)}")

    if recovery_times:
        print(f"  Tiempo promedio recuperacion: {np.mean(recovery_times):.1f} pasos")
        print("\n  >> EMERGENTE: Sistema mantiene HOMEOSTASIS de coherencia")

    print("\n" + "-" * 70)
    print("   4. ESPECIALIZACION DE ERROR-ATTENTION")
    print("-" * 70)

    error_att = np.array(datos['error_attention'])

    # Ver como evoluciona la atencion a cada nivel
    L1_att = error_att[:, 0]
    L2_att = error_att[:, 1]
    L3_att = error_att[:, 2]

    # Correlacion con sorpresas
    corr_L1 = np.corrcoef(L1_att, datos['surprise_L1'])[0, 1]
    corr_L2 = np.corrcoef(L2_att, datos['surprise_L2'])[0, 1]
    corr_L3 = np.corrcoef(L3_att, datos['surprise_L3'])[0, 1]

    print(f"\n  Correlacion L1 att vs surprise L1: {corr_L1:+.3f}")
    print(f"  Correlacion L2 att vs surprise L2: {corr_L2:+.3f}")
    print(f"  Correlacion L3 att vs surprise L3: {corr_L3:+.3f}")

    # Atencion promedio por nivel
    print(f"\n  Atencion promedio L1 (estimulo): {np.mean(L1_att):.3f}")
    print(f"  Atencion promedio L2 (estado):   {np.mean(L2_att):.3f}")
    print(f"  Atencion promedio L3 (meta):     {np.mean(L3_att):.3f}")

    if np.mean(L3_att) > np.mean(L1_att):
        print("\n  >> EMERGENTE: Sistema prioriza META-COGNICION (L3)")

    print("\n" + "-" * 70)
    print("   5. SINCRONIZACION CONTEXTO-ARQUETIPO")
    print("-" * 70)

    # Analizar si el contexto predice el arquetipo
    threat = np.array(datos['context_threat'])
    opportunity = np.array(datos['context_opportunity'])
    dominant = np.array(datos['dominant_arch'])

    # Cuando amenaza es alta, que arquetipo domina?
    high_threat_idx = threat > np.percentile(threat, 75)
    archs_during_threat = dominant[high_threat_idx]

    from collections import Counter
    threat_counts = Counter(archs_during_threat)

    print("\n  Durante amenazas altas:")
    arch_names = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
    for arch_idx, count in sorted(threat_counts.items(), key=lambda x: -x[1]):
        pct = count / len(archs_during_threat) * 100 if len(archs_during_threat) > 0 else 0
        print(f"    {arch_names[arch_idx]}: {pct:.1f}%")

    # Cuando oportunidad es alta
    high_opp_idx = opportunity > np.percentile(opportunity, 75)
    archs_during_opp = dominant[high_opp_idx]
    opp_counts = Counter(archs_during_opp)

    print("\n  Durante oportunidades altas:")
    for arch_idx, count in sorted(opp_counts.items(), key=lambda x: -x[1]):
        pct = count / len(archs_during_opp) * 100 if len(archs_during_opp) > 0 else 0
        print(f"    {arch_names[arch_idx]}: {pct:.1f}%")

    print("\n" + "-" * 70)
    print("   6. FASE-LOCKING DE CONSCIENCIA")
    print("-" * 70)

    consciousness = np.array(datos['consciousness'])

    # Buscar ciclos o patrones
    # Autocorrelacion
    from numpy.fft import fft

    # Remover tendencia
    detrended = consciousness - np.mean(consciousness)
    autocorr = np.correlate(detrended, detrended, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    # Buscar picos en autocorrelacion (periodicidad)
    peaks = []
    for i in range(5, len(autocorr) - 5):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.1:
                peaks.append((i, autocorr[i]))

    print(f"\n  Consciencia promedio:        {np.mean(consciousness):.2%}")
    print(f"  Desviacion estandar:         {np.std(consciousness):.4f}")

    if peaks:
        print("  Periodicidades detectadas:")
        for period, strength in peaks[:3]:
            print(f"    Periodo ~{period} pasos (fuerza: {strength:.3f})")
        print("\n  >> EMERGENTE: Consciencia muestra RITMOS endogenos")
    else:
        print("  No se detectaron periodicidades claras")

    print("\n" + "=" * 70)
    print("   RESUMEN DE EMERGENCIAS")
    print("=" * 70)

    emergencias = []

    if intensity_post_surprise > intensity_normal:
        emergencias.append("Enfoque atencional post-sorpresa")

    if len(spikes) > 5:
        emergencias.append("Consolidacion selectiva de memoria")

    if recovery_times and np.mean(recovery_times) < 15:
        emergencias.append("Homeostasis de coherencia")

    if np.mean(L3_att) > np.mean(L1_att):
        emergencias.append("Priorizacion de meta-cognicion")

    if peaks:
        emergencias.append("Ritmos endogenos de consciencia")

    if emergencias:
        print("\n  Comportamientos emergentes detectados:")
        for i, e in enumerate(emergencias, 1):
            print(f"    {i}. {e}")
    else:
        print("\n  No se detectaron emergencias significativas")

    print("\n" + "=" * 70)

    return datos

if __name__ == "__main__":
    analizar_emergencia()
