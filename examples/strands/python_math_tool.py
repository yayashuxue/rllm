"""Python code interpreter tool for Strands agent."""

import asyncio
from typing import Any, Dict

from rllm.tools.code_tools.python_interpreter import PythonInterpreter


class PythonMathTool:
    """A Python code interpreter tool adapted for Strands agent."""
    
    def __init__(self, backend="local"):
        """
        Initialize the Python tool.
        
        Args:
            backend: Backend type for the Python interpreter
        """
        self.interpreter = PythonInterpreter(backend=backend)
        self.name = "python"
        self.description = (
            "Execute Python code in a sandboxed environment. "
            "Use this to perform calculations, solve equations, create plots, "
            "and run mathematical computations. Always print your final answer."
        )
    
    def __call__(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python code and return results.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results
        """
        try:
            result = self.interpreter.forward(code, timeout=timeout)
            
            # Format the output for Strands
            output = {
                "success": True,
                "output": result.output or "",
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "error": result.error or ""
            }
            
            # Combine all output for display
            display_output = ""
            if result.stdout:
                display_output += f"Output:\n{result.stdout}\n"
            if result.output:
                display_output += f"Result: {result.output}\n"
            if result.stderr:
                display_output += f"Errors:\n{result.stderr}\n"
            if result.error:
                display_output += f"Error: {result.error}\n"
                
            output["display"] = display_output.strip()
            
            return output
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "display": f"Error executing code: {str(e)}"
            }
    
    async def __call_async__(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Async version of the call method."""
        # Run in executor since the underlying interpreter is sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.__call__, code, timeout)
    
    @property
    def json(self) -> Dict[str, Any]:
        """Return tool specification in Strands format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string", 
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (default: 30)",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        }


# Create a global instance for easy import
python_tool = PythonMathTool() 