#! /usr/bin/env python3
import rclpy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from simple_node import Node
from yasmin import State 
from yasmin import StateMachine

class SearchState(State):
    def __init__(self, state_publisher, blackboard):
        super().__init__(outcomes=['target_found', 'target_not_found'])
        self.state_publisher = state_publisher
        self.blackboard = blackboard

    def execute(self, blackboard):
        state_msg = String()
        state_msg.data = self.__class__.__name__
        self.state_publisher.publish(state_msg)

        if blackboard.get('target_found', False):
            return 'target_found'
        return 'target_not_found'

class ApproachState(State):
    def __init__(self, goal_publisher, state_publisher, blackboard):
        super().__init__(outcomes=['arrived', 'not_arrived'])
        self.goal_publisher = goal_publisher
        self.state_publisher = state_publisher
        self.blackboard = blackboard

    def execute(self, blackboard):
        state_msg = String()
        state_msg.data = self.__class__.__name__
        self.state_publisher.publish(state_msg)

        target_location = blackboard.get('target_location', None)
        if target_location is not None:
            self.goal_publisher.publish(target_location)
        
        if blackboard.get('goal_reached', False):
            return 'arrived'
        return 'not_arrived'

class QueryState(State):
    def __init__(self, state_publisher, blackboard):
        super().__init__(outcomes=['query_complete', 'query_not_complete'])
        self.state_publisher = state_publisher
        self.blackboard = blackboard

    def execute(self, blackboard):
        state_msg = String()
        state_msg.data = self.__class__.__name__
        self.state_publisher.publish(state_msg)

        if blackboard.get('query_completed', False):
            return 'query_complete'
        return 'query_not_complete'

class DoneState(State):
    def __init__(self, state_publisher, blackboard):
        super().__init__(outcomes=['done'])
        self.state_publisher = state_publisher
        self.blackboard = blackboard

    def execute(self, blackboard):
        state_msg = String()
        state_msg.data = self.__class__.__name__
        self.state_publisher.publish(state_msg)
        print('Searchstate blackboard: ', blackboard)
        
        blackboard['target_found'] = False
        blackboard['goal_reached'] = False
        blackboard['query_completed'] = False
        return 'done'
    
class FSM(Node):
    def __init__(self):
        super().__init__('trailbot_fsm')
        self.target_pose = None
        self.current_pose = None

        self.blackboard = {}

        self.state_publisher = self.create_publisher(String, 'fsm_state', 10) 

        self.target_subscription = self.create_subscription(
            PoseStamped,
            'target_pose',
            self.target_callback,
            10)

        self.amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.amcl_callback,
            10)

        self.path_subscription = self.create_subscription(
            String,
            'path_goal',
            10) 




def main(args=None):
    rclpy.init(args=args)
    node = FSM()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
