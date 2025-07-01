#!/usr/bin/env python3
"""
ANACOG (Agentic Cognitive Grammar Network) Demo
===============================================

This demo showcases the capabilities of the ANACOG system integrated into
the transformers library. ANACOG implements a distributed network of cognitive
grammars with hypergraph representations and neural-symbolic processing.

Example usage:
    python examples/anacog_demo.py
"""

import numpy as np

from transformers.integrations.anacog import (
    CognitiveHypergraph,
    HypergraphEdge,
    HypergraphNode,
    TensorShape,
    create_anacog_system,
    get_prime_tensor_shapes,
)


def main():
    print("🧠 ANACOG (Agentic Cognitive Grammar Network) Demo")
    print("=" * 55)

    # 1. Create ANACOG system
    print("\n1. Creating ANACOG System...")
    config = {"memory_capacity": 1000, "task_queue_size": 50, "adaptation_rate": 0.1}
    anacog = create_anacog_system(config)
    print(f"✅ ANACOG system initialized with config: {config}")

    # 2. Demonstrate tensor shapes and prime factorization
    print("\n2. Prime Kernel Tensor Shapes...")
    shapes = get_prime_tensor_shapes()
    for name, shape_info in shapes.items():
        print(f"  📊 {name}: {shape_info['shape']} (interaction: {shape_info['interaction_strength']})")

    # Create custom tensor shape
    custom_shape = TensorShape((128, 64, 32))
    print(f"\n  🔧 Custom tensor shape: {custom_shape.dimensions}")
    print(f"     Prime factors: {custom_shape.prime_factors}")
    print(f"     Degrees of freedom: {custom_shape.degrees_of_freedom}")

    # 3. Demonstrate hypergraph connectivity
    print("\n3. Hypergraph Connectivity...")
    hypergraph = CognitiveHypergraph()

    # Add cognitive nodes
    nodes = [
        HypergraphNode("agent_1", "agent", {"role": "coordinator"}, 0.8),
        HypergraphNode("task_1", "task", {"priority": "high"}, 0.9),
        HypergraphNode("memory_1", "memory", {"type": "semantic"}, 0.6),
        HypergraphNode("cognitive_prim_1", "cognitive_primitive", {"function": "pattern_match"}, 0.7),
    ]

    for node in nodes:
        hypergraph.add_node(node)
        print(f"  🔗 Added node: {node.id} ({node.node_type})")

    # Add hyperedges
    edges = [
        HypergraphEdge("edge_1", ["agent_1", "task_1"], "coordination", 0.9),
        HypergraphEdge("edge_2", ["task_1", "memory_1"], "retrieval", 0.7),
        HypergraphEdge("edge_3", ["agent_1", "memory_1", "cognitive_prim_1"], "processing", 0.8),
    ]

    for edge in edges:
        hypergraph.add_edge(edge)
        print(f"  🌐 Added edge: {edge.id} connecting {edge.nodes}")

    # Propagate activation
    print("\n  🚀 Propagating activation through hypergraph...")
    initial_activations = {node_id: node.activation for node_id, node in hypergraph.nodes.items()}
    hypergraph.propagate_activation(iterations=3)
    final_activations = {node_id: node.activation for node_id, node in hypergraph.nodes.items()}

    print("     Initial → Final activations:")
    for node_id in initial_activations:
        print(f"     {node_id}: {initial_activations[node_id]:.3f} → {final_activations[node_id]:.3f}")

    # 4. Demonstrate memory subsystem
    print("\n4. Memory Subsystem Processing...")
    test_memories = [
        "The cat sat on the mat",
        "Neural networks learn patterns",
        "Cognitive architectures enable reasoning",
        [1, 2, 3, 5, 8, 13],  # Fibonacci sequence
        np.array([0.1, 0.5, 0.9, 0.2]),  # Numerical data
    ]

    for i, memory in enumerate(test_memories):
        result = anacog.process(memory, subsystem="memory")
        print(f"  💭 Memory {i + 1}: {str(memory)[:30]}...")
        print(f"     Semantic shape: {result['memory']['semantic_encoding_shape']}")
        print(f"     Episodic shape: {result['memory']['episodic_encoding_shape']}")

    # 5. Demonstrate task subsystem
    print("\n5. Task Subsystem Management...")
    tasks = [
        ("analyze_patterns", "Analyze incoming data patterns", 0.8),
        ("generate_response", "Generate appropriate response", 0.9),
        ("update_memory", "Update long-term memory", 0.6),
        ("self_monitor", "Monitor system performance", 0.7),
        ("adapt_behavior", "Adapt behavior based on feedback", 0.5),
    ]

    for task_id, description, priority in tasks:
        result = anacog.process(f"Task: {description}", subsystem="task")
        print(f"  📋 Added task: {task_id} (priority: {priority})")

    # Get task priorities
    task_result = anacog.process("dummy", subsystem="task")
    print(f"     Current prioritized tasks: {task_result['task']['prioritized_tasks']}")

    # 6. Demonstrate AI analytics
    print("\n6. AI Analytics Subsystem...")
    analytics_data = [
        "This is a complex sentence with multiple clauses and semantic relationships.",
        [1, 4, 9, 16, 25, 36],  # Perfect squares
        {"key1": "value1", "key2": [1, 2, 3]},
    ]

    for data in analytics_data:
        result = anacog.process(data, subsystem="analytics")
        print(f"  🔍 Analyzing: {str(data)[:40]}...")
        print(f"     Patterns found: {len(result['analytics']['patterns'])}")
        print(f"     Logic results: {result['analytics']['logic_results']}")

    # 7. Demonstrate autonomous adaptation
    print("\n7. Autonomous Adaptation...")
    result = anacog.process("adaptation trigger", subsystem="adaptation")
    adaptation = result["adaptation"]
    print(f"  🎯 Meta-learning adjustment: {adaptation['meta_learning']['learning_rate_adjustment']}")
    print(f"     Architecture changes: {adaptation['meta_learning']['architecture_changes']}")
    print(f"     Resource allocation: {adaptation['resource_allocation']}")
    print(f"     Self-inspection: {adaptation['self_inspection']['system_state']}")

    # 8. Final system status
    print("\n8. Final System Status...")
    status = anacog.get_system_status()
    print("  📊 Memory subsystem:")
    print(f"     Semantic nodes: {status['memory_subsystem']['semantic_nodes']}")
    print(f"     Episodic episodes: {status['memory_subsystem']['episodic_episodes']}")
    print("  📊 Task subsystem:")
    print(f"     Active tasks: {status['task_subsystem']['active_tasks']}")
    print("  📊 Cognitive graph:")
    print(f"     Total nodes: {status['cognitive_graph']['total_nodes']}")
    print(f"     Total edges: {status['cognitive_graph']['total_edges']}")

    print("\n🎉 ANACOG Demo completed successfully!")
    print("\nThe system demonstrates:")
    print("  ✅ Prime factorization tensor shapes for cognitive subsystems")
    print("  ✅ Hypergraph connectivity with activation propagation")
    print("  ✅ Distributed memory (semantic + episodic)")
    print("  ✅ Task prioritization and workflow management")
    print("  ✅ AI analytics with pattern matching and inference")
    print("  ✅ Autonomous adaptation and self-inspection")

    return anacog


if __name__ == "__main__":
    system = main()
