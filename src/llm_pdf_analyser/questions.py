"""
Definitions for the 17 categorical questions used to analyse each article.
Each question has: id, texto, opcoes (answer options),
tipo ('multi' for multi-select, 'single' for single-select), tem_outro (free-text "Other" field).

Question text and options remain in Portuguese — they are sent directly to the
model as part of the prompt and appear as column names in the output CSV.
"""

import re
import unicodedata

QUESTIONS = [
    {
        "id": "fruta",
        "texto": "Qual é a fruta/cultura analisada nesse artigo?",
        "opcoes": [
            "Banana",
            "Uva",
            "Maçã",
            "Manga",
            "Caju",
            "Abacaxi",
            "Citros",
            "Morango",
        ],
        "tipo": "multi",
        "tem_outro": True,
    },
    {
        "id": "problema",
        "texto": "Qual problema da fruticultura o artigo busca resolver?",
        "opcoes": [
            "Detecção de doenças/pragas",
            "Classificação de frutos (qualidade/maturação)",
            "Estimativa de produtividade",
            "Irrigação inteligente/manejo hídrico",
            "Estresse hídrico e/ou nutricional",
            "Automação de colheita",
            "Previsão climática",
            "Mapeamento e caracterização de lavoura (solo, topografia, cobertura vegetal)",
            "Monitoramento do estado vegetativo da cultura",
            "Suporte à tomada de decisão/planejamento de manejo",
            "Rastreabilidade e gestão de cadeia produtiva",
        ],
        "tipo": "multi",
        "tem_outro": True,
    },
    {
        "id": "area_cientifica",
        "texto": "Em qual área científica da agricultura o trabalho se encontra?",
        "opcoes": [
            "Ciência do solo",
            "Fitotecnia",
            "Fisiologia Vegetal",
            "Melhoramento Vegetal e Genética",
            "Ciência das Plantas Daninhas",
            "Tecnologia de Sementes",
            "Proteção de plantas",
            "Logística/comercialização",
        ],
        "tipo": "multi",
        "tem_outro": False,
    },
    {
        "id": "algoritmo",
        "texto": "Qual algoritmo/modelo específico é utilizado?",
        "opcoes": [
            "CNN (Rede Neural Convolucional)",
            "YOLO (detecção de objetos em tempo real)",
            "ResNet / VGG / EfficientNet (CNNs pré-treinadas / transfer learning)",
            "Vision Transformer (ViT) / Swin Transformer",
            "Transformer / BERT / modelos de linguagem",
            "LSTM / GRU / Redes Neurais Recorrentes",
            "Autoencoder / Redes Generativas (GAN, VAE)",
            "Redes Neurais (MLP genérico / não especificado)",
            "SVM (Máquinas de Suporte Vetorial)",
            "Random Forest",
            "Gradient Boosting (XGBoost, LightGBM, CatBoost)",
            "Árvores de decisão",
            "k-NN (k-Vizinhos mais Próximos)",
            "Regressão (linear, logística, polinomial, etc.)",
            "Redes Bayesianas / Naive Bayes",
            "Lógica fuzzy",
            "Algoritmos genéticos / outras abordagens evolucionárias",
            "Clustering (K-means, DBSCAN, hierárquico, etc.)",
        ],
        "tipo": "multi",
        "tem_outro": True,
    },
    {
        "id": "tipo_dado",
        "texto": "Qual(is) tipo(s) de dado é(são) utilizado(s)?",
        "opcoes": [
            "Imagens (câmeras, drones, smartphones)",
            "Sensores de IoT (temperatura, umidade, etc.)",
            "Dados climáticos/meteorológicos",
            "Dados de solo",
            "Dados históricos/séries temporais",
            "Multimodal (combinação dos acima)",
        ],
        "tipo": "multi",
        "tem_outro": True,
    },
    {
        "id": "origem_dados",
        "texto": "Qual é a origem/contexto dos dados?",
        "opcoes": [
            "Campo real (produção agrícola)",
            "Estufa/ambiente controlado",
            "Dataset público disponível",
            "Simulação/dados sintéticos",
            "Laboratório",
        ],
        "tipo": "multi",
        "tem_outro": True,
    },
    {
        "id": "validacao_campo",
        "texto": "O modelo foi validado em ambiente real (campo)?",
        "opcoes": ["Sim", "Não", "Parcialmente"],
        "tipo": "single",
        "tem_outro": False,
    },
    {
        "id": "metrica",
        "texto": "Qual métrica principal de desempenho foi reportada?",
        "opcoes": [
            "Acurácia",
            "Precisão",
            "Recall/Sensibilidade",
            "F1-score",
            "RMSE (erro quadrático médio)",
            "MAE (erro absoluto médio)",
        ],
        "tipo": "multi",
        "tem_outro": True,
    },
    {
        "id": "faixa_resultado",
        "texto": "Qual é a faixa do resultado reportado?",
        "opcoes": ["menor que 60%", "60% a 80%", "80% a 100%"],
        "tipo": "single",
        "tem_outro": False,
    },
    {
        "id": "compara_baseline",
        "texto": "O estudo compara a IA com métodos tradicionais/baseline?",
        "opcoes": ["Sim", "Não"],
        "tipo": "single",
        "tem_outro": False,
    },
    {
        "id": "estagio",
        "texto": "Em que estágio a solução se encontra?",
        "opcoes": [
            "Conceitual/teórico",
            "Experimental (laboratório)",
            "Protótipo (testado em pequena escala)",
            "Testado em campo (validação agrícola)",
            "Implementação comercial/uso real",
        ],
        "tipo": "single",
        "tem_outro": False,
    },
    {
        "id": "publico_alvo",
        "texto": "Qual é o público-alvo da solução?",
        "opcoes": [
            "Pequeno produtor/agricultura familiar",
            "Médio produtor",
            "Agronegócio em larga escala",
            "Todos os públicos",
        ],
        "tipo": "single",
        "tem_outro": False,
    },
    {
        "id": "infraestrutura",
        "texto": "A solução depende de infraestrutura específica?",
        "opcoes": [
            "Internet/conectividade",
            "Hardware especializado (GPU, drones, etc.)",
            "Sensores específicos de alto custo",
            "Conhecimento técnico avançado",
            "Nenhuma dependência relevante",
        ],
        "tipo": "multi",
        "tem_outro": False,
    },
    {
        "id": "custo",
        "texto": "Qual é o nível de custo/viabilidade econômica da solução?",
        "opcoes": [
            "Baixo custo (acessível para pequenos produtores)",
            "Custo médio (viável para produtores médios)",
            "Alto custo (apenas grandes operações)",
            "Não mencionado",
        ],
        "tipo": "single",
        "tem_outro": False,
    },
    {
        "id": "menciona_limitacoes",
        "texto": "O artigo menciona limitações ou desafios?",
        "opcoes": ["Sim", "Não"],
        "tipo": "single",
        "tem_outro": False,
    },
    {
        "id": "limitacoes",
        "texto": "Se sim, quais são as principais limitações/desafios mencionados?",
        "opcoes": [
            "Falta/qualidade de dados",
            "Baixa generalização/replicabilidade",
            "Alto custo de implementação",
            "Dependência de infraestrutura",
            "Escalabilidade limitada",
            "Falta de validação em campo",
        ],
        "tipo": "multi",
        "tem_outro": True,
    },
    {
        "id": "beneficios",
        "texto": "Qual(is) benefício(s) operacional(is) a IA proporciona?",
        "opcoes": [
            "Aumento de produtividade",
            "Redução de custos operacionais",
            "Otimização de insumos (água, fertilizantes)",
            "Sustentabilidade ambiental",
            "Automação de processos",
            "Tomada de decisão em tempo real",
        ],
        "tipo": "multi",
        "tem_outro": True,
    },
]


def get_all_csv_columns() -> list[str]:
    """
    Returns all column names for the output CSV, in order.
    Useful for initialising the CSV header and ensuring consistency.
    """
    cols = ["arquivo", "status"]
    for q in QUESTIONS:
        q_id = q["id"]
        if q["tipo"] == "single":
            for opt in q["opcoes"]:
                cols.append(_make_col_name(q_id, opt))
        else:
            for opt in q["opcoes"]:
                cols.append(_make_col_name(q_id, opt))
            if q.get("tem_outro"):
                cols.append(f"{q_id}_outro_texto")
    return cols


def _make_col_name(question_id: str, option: str) -> str:
    """Generates a normalised column name from a question id and option text."""
    slug = option.lower()
    # Remove parenthetical content to keep column names short
    slug = re.sub(r"\(.*?\)", "", slug)
    slug = slug.strip()
    # Decompose Unicode accents (NFD) and strip diacritic combining characters
    slug = unicodedata.normalize("NFD", slug)
    slug = "".join(c for c in slug if unicodedata.category(c) != "Mn")
    # Replace non-alphanumeric characters with underscores
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return f"{question_id}__{slug}"
