# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from transformers.integrations.anacog import (
    AIAnalyticsSubsystem,
    ANACOGHub,
    AutonomousAdaptationSubsystem,
    CognitiveHypergraph,
    HypergraphEdge,
    HypergraphNode,
    MemorySubsystem,
    TaskSubsystem,
    TensorShape,
    create_anacog_system,
    get_prime_tensor_shapes,
)
from transformers.testing_utils import TestCasePlus


class ANACOGIntegrationTests(TestCasePlus):
    """Test cases for ANACOG (Agentic Cognitive Grammar Network) integration."""

    def test_tensor_shape_creation(self):
        """Test creation and prime factorization of tensor shapes."""
        shape = TensorShape((64, 32, 16))

        self.assertEqual(shape.dimensions, (64, 32, 16))
        self.assertEqual(shape.degrees_of_freedom, 3)

        # Check prime factorization
        self.assertIn(0, shape.prime_factors)
        self.assertEqual(shape.prime_factors[0], [2, 2, 2, 2, 2, 2])  # 64 = 2^6
        self.assertEqual(shape.prime_factors[1], [2, 2, 2, 2, 2])  # 32 = 2^5
        self.assertEqual(shape.prime_factors[2], [2, 2, 2, 2])  # 16 = 2^4

        # Check interaction space
        self.assertIsNotNone(shape.interaction_space)
        self.assertEqual(shape.interaction_space.shape, (3, 3))

    def test_cognitive_hypergraph(self):
        """Test hypergraph creation and manipulation."""
        graph = CognitiveHypergraph()

        # Add nodes
        node1 = HypergraphNode(id="node1", node_type="agent", activation=0.5)
        node2 = HypergraphNode(id="node2", node_type="task", activation=0.7)
        graph.add_node(node1)
        graph.add_node(node2)

        self.assertEqual(len(graph.nodes), 2)

        # Add edge
        edge = HypergraphEdge(id="edge1", nodes=["node1", "node2"], edge_type="connection", weight=0.8)
        graph.add_edge(edge)

        self.assertEqual(len(graph.edges), 1)
        self.assertIsNotNone(graph.adjacency_tensor)

        # Test activation propagation
        initial_activation1 = graph.nodes["node1"].activation
        initial_activation2 = graph.nodes["node2"].activation

        graph.propagate_activation(iterations=1)

        # Activations should have changed
        self.assertNotEqual(graph.nodes["node1"].activation, initial_activation1)
        self.assertNotEqual(graph.nodes["node2"].activation, initial_activation2)

    def test_memory_subsystem(self):
        """Test memory subsystem functionality."""
        memory = MemorySubsystem()

        # Test semantic memory
        test_data = "Hello, this is a test semantic memory"
        semantic_result = memory.encode_memory(test_data, "semantic")

        self.assertIsInstance(semantic_result, np.ndarray)
        self.assertTrue(len(memory.semantic_memory.embeddings) > 0)

        # Test episodic memory
        episodic_result = memory.encode_memory([1, 2, 3, 4, 5], "episodic")

        self.assertIsInstance(episodic_result, np.ndarray)
        self.assertTrue(len(memory.episodic_cache.episodes) > 0)

    def test_task_subsystem(self):
        """Test task subsystem functionality."""
        task_system = TaskSubsystem()

        # Add tasks
        task_system.add_task("task1", "Process input data", priority=0.8)
        task_system.add_task("task2", "Analyze patterns", priority=0.6)
        task_system.add_task("task3", "Generate output", priority=0.9)

        self.assertEqual(len(task_system.task_graph.nodes), 3)

        # Test prioritization
        prioritized = task_system.prioritize_tasks()

        self.assertEqual(len(prioritized), 3)
        # Highest priority task should be first
        self.assertEqual(prioritized[0], "task3")  # priority 0.9

    def test_ai_analytics_subsystem(self):
        """Test AI analytics subsystem."""
        analytics = AIAnalyticsSubsystem()

        # Test string analysis
        result = analytics.analyze("test string data")

        self.assertIn("patterns", result)
        self.assertIn("logic_results", result)
        self.assertIn("optimized_results", result)

        # Test sequence analysis
        result = analytics.analyze([1, 2, 3, 4, 5])

        self.assertIn("patterns", result)
        self.assertTrue(len(result["patterns"]) > 0)

    def test_autonomous_adaptation_subsystem(self):
        """Test autonomous adaptation subsystem."""
        adaptation = AutonomousAdaptationSubsystem()

        performance_metrics = {"memory_efficiency": 0.7, "task_completion": 0.5, "pattern_recognition": 0.8}

        result = adaptation.adapt(performance_metrics)

        self.assertIn("meta_learning", result)
        self.assertIn("resource_allocation", result)
        self.assertIn("self_inspection", result)

        # Check resource allocation logic
        allocation = result["resource_allocation"]
        self.assertIsInstance(allocation, dict)
        # Lower performing tasks should get more resources
        self.assertTrue(allocation["task_completion"] > allocation["pattern_recognition"])

    def test_anacog_hub_integration(self):
        """Test the main ANACOG hub integration."""
        hub = ANACOGHub()

        # Test processing through all subsystems
        test_input = "Integration test data"
        results = hub.process(test_input, subsystem="all")

        self.assertIn("memory", results)
        self.assertIn("task", results)
        self.assertIn("analytics", results)
        self.assertIn("adaptation", results)

        # Test individual subsystem processing
        memory_result = hub.process(test_input, subsystem="memory")
        self.assertIn("memory", memory_result)
        self.assertNotIn("task", memory_result)

        # Test system status
        status = hub.get_system_status()
        self.assertIn("memory_subsystem", status)
        self.assertIn("task_subsystem", status)
        self.assertIn("cognitive_graph", status)

    def test_factory_function(self):
        """Test the factory function for creating ANACOG systems."""
        system1 = create_anacog_system()
        self.assertIsInstance(system1, ANACOGHub)

        config = {"test_param": "test_value"}
        system2 = create_anacog_system(config)
        self.assertEqual(system2.config["test_param"], "test_value")

    def test_prime_tensor_shapes(self):
        """Test the predefined prime tensor shapes."""
        shapes = get_prime_tensor_shapes()

        self.assertIn("memory_semantic", shapes)
        self.assertIn("memory_episodic", shapes)
        self.assertIn("task_priority", shapes)
        self.assertIn("analytics_pattern", shapes)
        self.assertIn("adaptation_meta", shapes)

        # Verify semantic memory shape
        semantic_shape = shapes["memory_semantic"]
        self.assertEqual(semantic_shape["shape"], (512, 256, 128))
        self.assertEqual(semantic_shape["interaction_strength"], "high")

    def test_hypergraph_error_handling(self):
        """Test error handling in hypergraph operations."""
        graph = CognitiveHypergraph()

        # Try to add edge with non-existent nodes
        edge = HypergraphEdge(id="edge1", nodes=["nonexistent1", "nonexistent2"], edge_type="connection")

        with self.assertRaises(ValueError):
            graph.add_edge(edge)

    def test_tensor_shape_edge_cases(self):
        """Test tensor shape with edge cases."""
        # Empty dimensions
        empty_shape = TensorShape(())
        self.assertEqual(empty_shape.degrees_of_freedom, 0)
        self.assertEqual(len(empty_shape.interaction_space), 0)

        # Single dimension
        single_shape = TensorShape((8,))
        self.assertEqual(single_shape.degrees_of_freedom, 1)
        self.assertEqual(single_shape.prime_factors[0], [2, 2, 2])  # 8 = 2^3

    def test_memory_subsystem_different_inputs(self):
        """Test memory subsystem with various input types."""
        memory = MemorySubsystem()

        # Test with different data types
        string_data = "Test string"
        list_data = [1, 2, 3, 4]
        array_data = np.array([1.0, 2.0, 3.0])

        string_result = memory.encode_memory(string_data, "semantic")
        list_result = memory.encode_memory(list_data, "episodic")
        array_result = memory.encode_memory(array_data, "semantic")

        self.assertIsInstance(string_result, np.ndarray)
        self.assertIsInstance(list_result, np.ndarray)
        self.assertIsInstance(array_result, np.ndarray)

    def test_workflow_parsing(self):
        """Test workflow graph parsing."""
        task_system = TaskSubsystem()

        workflow_spec = {
            "nodes": [
                {"id": "node1", "attributes": {"type": "start"}, "activation": 1.0},
                {"id": "node2", "attributes": {"type": "process"}, "activation": 0.5},
            ],
            "edges": [{"id": "edge1", "nodes": ["node1", "node2"], "weight": 0.8}],
        }

        workflow_graph = task_system.workflow_parser.parse_workflow(workflow_spec)

        self.assertEqual(len(workflow_graph.nodes), 2)
        self.assertEqual(len(workflow_graph.edges), 1)
        self.assertEqual(workflow_graph.nodes["node1"].activation, 1.0)
