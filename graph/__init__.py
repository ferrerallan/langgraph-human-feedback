"""
Pacote para definição e construção do grafo de fluxo.
"""
# Primeiro, importe explicitamente a classe State
from graph.state import State

# Em seguida, importe o construtor do grafo
from graph.builder import GraphBuilder

# Exporte as classes para que possam ser importadas de outros módulos
__all__ = ['State', 'GraphBuilder']