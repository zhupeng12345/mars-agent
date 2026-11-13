"""JEV-facing protocol facade.

The canonical contracts live in ``agentic.platform.contracts`` so all agents
share one schema.  This module provides the stable JEV import boundary used by
adapters and downstream integrations without creating a second protocol set.
"""

from agentic.platform.contracts import InspectionFeatures, JevRouteDecision, Route

__all__ = ["InspectionFeatures", "JevRouteDecision", "Route"]
