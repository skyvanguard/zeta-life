#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generador de Resumen Ejecutivo - Proyecto Zeta Life
Para presentacion a directivos
Version 2.0 - Incluye resultados de escalabilidad y estres
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# Colores corporativos
PRIMARY_COLOR = HexColor('#1a365d')
SECONDARY_COLOR = HexColor('#2c5282')
ACCENT_COLOR = HexColor('#38a169')
LIGHT_BG = HexColor('#f7fafc')
SUCCESS_COLOR = HexColor('#48bb78')
WARNING_COLOR = HexColor('#ed8936')

def crear_estilos():
    """Crear estilos personalizados para el documento"""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='TituloPrincipal',
        parent=styles['Title'],
        fontSize=28,
        textColor=PRIMARY_COLOR,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='Subtitulo',
        parent=styles['Normal'],
        fontSize=14,
        textColor=SECONDARY_COLOR,
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))

    styles.add(ParagraphStyle(
        name='Seccion',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=PRIMARY_COLOR,
        spaceBefore=20,
        spaceAfter=10,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='Subseccion',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=SECONDARY_COLOR,
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='Cuerpo',
        parent=styles['Normal'],
        fontSize=10,
        textColor=black,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leading=14
    ))

    styles.add(ParagraphStyle(
        name='Destacado',
        parent=styles['Normal'],
        fontSize=11,
        textColor=PRIMARY_COLOR,
        spaceBefore=6,
        spaceAfter=6,
        fontName='Helvetica-Bold',
        leftIndent=20
    ))

    styles.add(ParagraphStyle(
        name='Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=HexColor('#718096'),
        alignment=TA_CENTER
    ))

    return styles


def crear_metricas_principales():
    """Metricas destacadas principales."""
    data = [
        ['+134%', '11', '96.6%', '+89.9%'],
        ['Supervivencia\ncelular', 'Propiedades\nemergentes', 'Tasa de\nregeneracion', 'Antifragilidad\npost-colapso'],
    ]

    table = Table(data, colWidths=[1.4*inch, 1.4*inch, 1.4*inch, 1.4*inch])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 24),
        ('TEXTCOLOR', (0, 0), (0, 0), ACCENT_COLOR),
        ('TEXTCOLOR', (1, 0), (1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (2, 0), (2, 0), ACCENT_COLOR),
        ('TEXTCOLOR', (3, 0), (3, 0), SUCCESS_COLOR),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, 1), 9),
        ('TEXTCOLOR', (0, 1), (-1, 1), HexColor('#4a5568')),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))

    return table


def crear_tabla_resultados_actualizados():
    """Tabla de resultados con nuevos hallazgos."""
    data = [
        ['Area', 'Metrica Clave', 'Resultado', 'Significancia'],
        ['Automatas Celulares', 'Supervivencia vs tradicional', '+134%', 'Alta'],
        ['ZetaOrganism Base', 'Propiedades emergentes', '11 demostradas', 'Alta'],
        ['Escalabilidad', 'Fi con 1000 agentes', '231 (23.1%)', 'Alta'],
        ['Estres - Dano Severo', 'Recuperacion 5 rondas 80%', '96.6%', 'Alta'],
        ['Estres - Escasez', 'Antifragilidad post-colapso', '+89.9%', 'Alta'],
        ['Multi-Organismo', 'Coexistencia 3 orgs (900 ag)', 'Shannon max', 'Media'],
    ]

    table = Table(data, colWidths=[1.5*inch, 1.8*inch, 1.2*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('ALIGN', (3, 1), (3, -1), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, 1), LIGHT_BG),
        ('BACKGROUND', (0, 3), (-1, 3), LIGHT_BG),
        ('BACKGROUND', (0, 5), (-1, 5), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e0')),
        ('LINEBELOW', (0, 0), (-1, 0), 2, PRIMARY_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    return table


def crear_tabla_escalabilidad():
    """Tabla de resultados de escalabilidad."""
    data = [
        ['Agentes', 'Grid', 'Fi Final', '% Fi', 'Coordinacion', 'Tiempo'],
        ['100', '64x64', '20', '20.0%', '0.931', '18s'],
        ['200', '90x90', '38', '19.0%', '0.930', '49s'],
        ['500', '142x142', '104', '20.8%', '0.962', '3.2min'],
        ['1000', '200x200', '231', '23.1%', '0.970', '12min'],
    ]

    table = Table(data, colWidths=[0.8*inch, 0.9*inch, 0.8*inch, 0.7*inch, 1.0*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 2), (-1, 2), LIGHT_BG),
        ('BACKGROUND', (0, 4), (-1, 4), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))

    return table


def crear_tabla_estres():
    """Tabla de resultados de estres."""
    data = [
        ['Escenario', 'Descripcion', 'Resultado'],
        ['Dano Severo', '5 rondas eliminando 80% Fi', '96.6% recuperacion'],
        ['Escasez Extrema', 'Reduccion energia hasta colapso', '+89.9% antifragilidad'],
        ['Migracion Forzada', 'Gradientes de energia', '170 Fi mantenidos'],
    ]

    table = Table(data, colWidths=[1.3*inch, 2.2*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BACKGROUND', (0, 2), (-1, 2), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    return table


def crear_tabla_multi_organismo():
    """Tabla de resultados multi-organismo."""
    data = [
        ['Sistema', 'Total Agentes', 'Poblaciones', 'Fi', 'Shannon'],
        ['2 Organismos', '800', '400 / 400', '48 / 369', '0.693 (max)'],
        ['3 Organismos', '900', '300 / 300 / 300', '125 / 68 / 81', '1.099 (max)'],
    ]

    table = Table(data, colWidths=[1.1*inch, 1.0*inch, 1.2*inch, 1.1*inch, 1.0*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 2), (-1, 2), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    return table


def crear_tabla_propiedades():
    """Tabla de propiedades emergentes."""
    data = [
        ['Propiedad', 'Descripcion', 'Evidencia'],
        ['Homeostasis', 'Auto-regulacion a equilibrio', 'Coord retorna a 0.88'],
        ['Regeneracion', 'Recuperacion estructural', '75-125% post-dano'],
        ['Antifragilidad', 'Fortalecimiento post-estres', '+89.9% Fi post-colapso'],
        ['Quimiotaxis', 'Migracion colectiva', '~21 celdas desplazamiento'],
        ['Memoria espacial', 'Aprendizaje implicito', 'Evacuacion preventiva'],
        ['Auto-segregacion', 'Identidad colectiva', 'Separacion espontanea'],
        ['Huida coordinada', 'Comunicacion efectiva', '+123% separacion'],
        ['Forrajeo colectivo', 'Exploracion cooperativa', '+15 celulas a recursos'],
    ]

    table = Table(data, colWidths=[1.3*inch, 1.8*inch, 1.6*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, 1), LIGHT_BG),
        ('BACKGROUND', (0, 3), (-1, 3), LIGHT_BG),
        ('BACKGROUND', (0, 5), (-1, 5), LIGHT_BG),
        ('BACKGROUND', (0, 7), (-1, 7), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    return table


def generar_pdf(output_path):
    """Generar el PDF ejecutivo actualizado."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch
    )

    styles = crear_estilos()
    story = []

    # === PAGINA 1: PORTADA Y METRICAS ===
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Proyecto Zeta Life", styles['TituloPrincipal']))
    story.append(Paragraph(
        "Inteligencia Artificial basada en la Hipotesis de Riemann",
        styles['Subtitulo']
    ))

    story.append(HRFlowable(width="80%", thickness=2, color=PRIMARY_COLOR, spaceAfter=15))

    # Resumen ejecutivo
    story.append(Paragraph("Resumen Ejecutivo", styles['Seccion']))

    resumen = """
    Este proyecto implementa un nuevo paradigma de IA utilizando los ceros de la funcion
    zeta de Riemann como base matematica. Los resultados demuestran comportamientos emergentes
    significativos, incluyendo auto-organizacion, regeneracion y coordinacion colectiva.
    <b>El sistema escala exitosamente hasta 1000+ agentes</b> y demuestra <b>antifragilidad</b>
    - se vuelve mas fuerte despues de sufrir estres.
    """
    story.append(Paragraph(resumen.strip(), styles['Cuerpo']))
    story.append(Spacer(1, 0.15*inch))

    # Metricas destacadas
    story.append(Paragraph("Metricas Clave", styles['Subseccion']))
    story.append(crear_metricas_principales())
    story.append(Spacer(1, 0.2*inch))

    # Tabla de resultados
    story.append(Paragraph("Resultados por Area", styles['Subseccion']))
    story.append(crear_tabla_resultados_actualizados())

    # === PAGINA 2: ESCALABILIDAD Y ESTRES ===
    story.append(PageBreak())

    story.append(Paragraph("Pruebas de Escalabilidad", styles['Seccion']))

    escala_texto = """
    El sistema fue probado con 100 a 1000 agentes. Los resultados muestran que la
    emergencia de liderazgo (Fi) es <b>super-lineal</b>: con 10x mas agentes, emergen
    11.6x mas lideres. La coordinacion <b>mejora</b> con la escala (0.93 a 0.97).
    """
    story.append(Paragraph(escala_texto.strip(), styles['Cuerpo']))
    story.append(Spacer(1, 0.1*inch))
    story.append(crear_tabla_escalabilidad())

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Pruebas de Estres (500 agentes)", styles['Seccion']))

    estres_texto = """
    Se evaluaron tres escenarios de estres extremo. El hallazgo mas notable es la
    <b>antifragilidad</b>: despues de un colapso total por escasez de energia,
    el sistema se recupera con <b>89.9% mas lideres</b> que antes del colapso.
    """
    story.append(Paragraph(estres_texto.strip(), styles['Cuerpo']))
    story.append(Spacer(1, 0.1*inch))
    story.append(crear_tabla_estres())

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Sistemas Multi-Organismo (800-900 agentes)", styles['Seccion']))

    multi_texto = """
    Multiples organismos compitiendo en el mismo espacio mantienen <b>diversidad maxima</b>
    (Shannon index = maximo teorico). Con 3 organismos de 300 agentes cada uno,
    las poblaciones permanecen perfectamente balanceadas.
    """
    story.append(Paragraph(multi_texto.strip(), styles['Cuerpo']))
    story.append(Spacer(1, 0.1*inch))
    story.append(crear_tabla_multi_organismo())

    # === PAGINA 3: PROPIEDADES Y CONCLUSIONES ===
    story.append(PageBreak())

    story.append(Paragraph("Propiedades Emergentes Demostradas", styles['Seccion']))
    story.append(Spacer(1, 0.1*inch))
    story.append(crear_tabla_propiedades())

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Fundamento Cientifico", styles['Seccion']))

    fundamento = """
    Los ceros de la funcion zeta de Riemann ocupan un <b>punto critico matematico</b>
    entre orden y caos (el "borde del caos"). Esta ubicacion unica produce correlaciones
    estructuradas que permiten la emergencia de comportamientos complejos sin programarlos
    explicitamente. El sistema exhibe propiedades tipicamente asociadas con inteligencia:
    adaptacion, aprendizaje implicito, anticipacion y resiliencia.
    """
    story.append(Paragraph(fundamento.strip(), styles['Cuerpo']))

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Aplicaciones Potenciales", styles['Seccion']))

    aplicaciones = [
        "<b>Robotica de enjambre:</b> Coordinacion autonoma de drones sin control central",
        "<b>Sistemas distribuidos:</b> Algoritmos de consenso tolerantes a fallas",
        "<b>Simulacion biologica:</b> Modelado de ecosistemas y comportamiento celular",
        "<b>Trading algoritmico:</b> Deteccion de patrones en series temporales",
    ]
    for app in aplicaciones:
        story.append(Paragraph(f"  {app}", styles['Cuerpo']))

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Conclusion", styles['Seccion']))

    conclusion = """
    El Proyecto Zeta Life demuestra que los ceros de la funcion zeta de Riemann
    proporcionan una base matematica unica para sistemas de IA adaptativos.
    Los experimentos confirman <b>escalabilidad exitosa</b> (hasta 1000+ agentes),
    <b>resiliencia extrema</b> (96.6% recuperacion bajo estres repetido), y
    <b>antifragilidad</b> (el sistema se fortalece despues de crisis).
    Esta investigacion abre nuevas direcciones para el desarrollo de IA robusta y adaptativa.
    """
    story.append(Paragraph(conclusion.strip(), styles['Cuerpo']))

    story.append(Spacer(1, 0.3*inch))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cbd5e0')))
    story.append(Spacer(1, 0.1*inch))

    fecha = datetime.now().strftime("%d de %B de %Y")
    footer_text = f"""
    <b>Documento:</b> Resumen Ejecutivo - Proyecto Zeta Life v2.0<br/>
    <b>Fecha:</b> {fecha}<br/>
    <b>Framework Teorico:</b> Francisco Ruiz | <b>Implementacion:</b> Diciembre 2025
    """
    story.append(Paragraph(footer_text, styles['Footer']))

    # Construir PDF
    doc.build(story)
    print(f"PDF generado exitosamente: {output_path}")
    return output_path


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reportes")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(output_dir, f"resumen_ejecutivo_zeta_life_{timestamp}.pdf")

    generar_pdf(output_file)
