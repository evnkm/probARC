import os
import json
from typing import Dict, List

from task_order import TaskOrder

print(TaskOrder.first)

class ConceptARCLoader:
    def __init__(self, phase: str = 'first', base_path: str = 'ConceptARCSmall'):
        self.base_path = base_path
        self.task_order = TaskOrder
        self.loaded_tasks = self.load_tasks_by_category(phase)

    def load_tasks_by_category(self, phase: str = 'first') -> Dict[str, Dict[int, Dict]]:
        """
        Load JSON tasks from ConceptARCSmall directory based on the specified task order.
        
        :param phase: Which phase of tasks to load ('first' or 'second')
        :return: Dictionary of tasks grouped by category and task number
        """
        task_order = getattr(self.task_order, phase)
        loaded_tasks = {}

        for category, task_files in task_order.items():
            category_path = os.path.join(self.base_path, category)
            loaded_tasks[category] = {}

            for task_file in task_files:
                # Extract task number from filename (e.g., 'AboveBelow2.json' -> 2)
                task_number = self.extract_task_number_from_file(task_file)
                full_path = os.path.join(category_path, task_file)
                try:
                    with open(full_path, 'r') as f:
                        task_data = json.load(f)
                        loaded_tasks[category][task_number] = {
                            'train': task_data.get('train', []),
                            'test': task_data.get('test', [])
                        }
                except FileNotFoundError:
                    print(f"Warning: Task file {full_path} not found.")
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON from {full_path}")

        return loaded_tasks

    def extract_task_number_from_file(self, task_file):
        try:
            task_number = int(''.join(filter(str.isdigit, task_file)))
            return task_number
        except ValueError:
            raise ValueError(f"Warning: Could not extract task number from {task_file}")

    def get_task_categories(self) -> List[str]:
        """
        Get the list of task categories.
        
        :return: List of task category names
        """
        return list(self.task_order.first.keys())

    def get_all_pairs(self, category: str, task_number: int) -> List:
        """
        Get the train and test data for a specific task.
        
        :param category: Task category
        :param task_number: Task number
        :return: List of task data (train + test)
        """
        return self.loaded_tasks[category][task_number]['train'] + self.loaded_tasks[category][task_number]['test']

# Example usage
if __name__ == '__main__':
    first_loader = ConceptARCLoader("first", "/om/user/evan_kim/966/probARC/ConceptARCSmall")
    second_loader = ConceptARCLoader("second", "/om/user/evan_kim/966/probARC/ConceptARCSmall")
    
    # Load first phase tasks
    first_phase_tasks = first_loader.loaded_tasks
    for category, tasks in first_phase_tasks.items():
        print(f"Category: {category}")
        for task_number, task_data in tasks.items():
            print(task_number, task_data)
            print(f"TASK {task_number}: Train + test:", len(task_data['train']) + len(task_data['test']) == len(first_loader.get_all_pairs(category, task_number)))
        print("check if tasks are different order:", tuple(first_loader.loaded_tasks[category].keys()) == tuple(second_loader.loaded_tasks[category].keys()))
        print(tuple(first_loader.loaded_tasks[category].keys()), tuple(second_loader.loaded_tasks[category].keys()))
    
    print("+====================================+")
    
    source_dir = "/om/user/evan_kim/966/probARC/ConceptARCSmall"
    sequential_common_arc_dir = "/om/user/evan_kim/966/probARC/ConceptARCSmallSequential"
    for category, tasks in first_phase_tasks.items():
        print(f"Category: {category}")
        # Create category directory if it doesn't exist
        category_dir = os.path.join(sequential_common_arc_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for task_number, task_data in tasks.items():
            total_examples = len(task_data['train']) + len(task_data['test'])
            
            # Create sequential files from 2 to total_examples
            for i in range(2, total_examples + 1):
                # Calculate how many examples go into train vs test
                train_count = i - 1  # All but one example goes to train
                test_count = 1       # One example always goes to test
                
                # Create new task data structure
                combined_tasks = task_data['train'] + task_data['test']
                new_task_data = {
                    'train': combined_tasks[:train_count],
                    'test': combined_tasks[train_count:train_count+test_count]
                }
                
                # Create filename (e.g., "AboveBelow2_3.json")
                base_filename = next(f for f in os.listdir(os.path.join(source_dir, category)) 
                                   if str(task_number) in f and f.endswith('.json'))
                new_filename = f"{os.path.splitext(base_filename)[0]}_{i}.json"
                
                # Save the new JSON file
                output_path = os.path.join(category_dir, new_filename)
                with open(output_path, 'w') as f:
                    json.dump(new_task_data, f, indent=2)
                
                print(f"Created {new_filename} with {train_count} train and {test_count} test examples")