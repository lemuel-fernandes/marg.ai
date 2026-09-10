"""Inter-module message bus."""

import logging
from typing import Callable, Any, Dict, List

logger = logging.getLogger(__name__)

class MessageBus:
    """
    A simple synchronous Pub/Sub message bus. 
    Allows modules to communicate without hardcoding dependencies.
    """
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, topic: str, callback: Callable):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        logger.debug(f"Subscribed to topic: {topic}")

    def publish(self, topic: str, data: Any):
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for topic {topic}: {e}")
        else:
            logger.warning(f"No subscribers for topic: {topic}")