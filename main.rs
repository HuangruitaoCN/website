use pathfinding::prelude::astar;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GridPosition {
    pub x: i32,
    pub y: i32,
}

impl GridPosition {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &GridPosition) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

#[derive(Debug, Clone)]
pub struct WarehouseGrid {
    pub width: i32,
    pub height: i32,
    pub obstacles: HashSet<GridPosition>,
    pub temporary_obstacles: HashMap<GridPosition, u32>,
}

impl WarehouseGrid {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            obstacles: HashSet::new(),
            temporary_obstacles: HashMap::new(),
        }
    }

    pub fn add_obstacle(&mut self, x: i32, y: i32) {
        self.obstacles.insert(GridPosition::new(x, y));
    }

    pub fn add_obstacle_pos(&mut self, pos: GridPosition) {
        self.obstacles.insert(pos);
    }

    pub fn is_accessible(&self, pos: &GridPosition, current_time: u32) -> bool {
        // Check bounds
        if pos.x < 0 || pos.x >= self.width || pos.y < 0 || pos.y >= self.height {
            return false;
        }
        
        // Check permanent obstacles
        if self.obstacles.contains(pos) {
            return false;
        }
        
        // Check temporary obstacles
        if let Some(&expiry) = self.temporary_obstacles.get(pos) {
            if expiry > current_time {
                return false;
            }
        }
        
        true
    }

    pub fn find_path(
        &self,
        start: GridPosition,
        goal: GridPosition,
        current_time: u32,
    ) -> Option<(Vec<GridPosition>, i32)> {
        astar(
            &start,
            |p| self.successors(p, current_time),
            |p| p.distance_to(&goal),
            |p| *p == goal,
        )
    }

    fn successors(&self, pos: &GridPosition, current_time: u32) -> Vec<(GridPosition, i32)> {
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]; // 4-direction movement
        let mut successors = Vec::new();

        for &(dx, dy) in &directions {
            let new_pos = GridPosition::new(pos.x + dx, pos.y + dy);

            if self.is_accessible(&new_pos, current_time) {
                successors.push((new_pos, 1)); // Cost of 1 for each move
            }
        }

        successors
    }

    pub fn update_temporary_obstacles(&mut self, current_time: u32) {
        self.temporary_obstacles.retain(|_, &mut expiry| expiry > current_time);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RobotState {
    Idle,
    Moving,
    WaitingForPath,
    Paused,
    Completed,
    Error,
}

#[derive(Debug, Clone)]
pub struct Robot {
    pub id: u32,
    pub name: String,
    pub position: GridPosition,
    pub target: GridPosition,
    pub path: VecDeque<GridPosition>,
    pub state: RobotState,
    pub paused_until: Option<u32>,
    pub last_update: u32,
}

impl Robot {
    pub fn new(id: u32, name: &str, start_x: i32, start_y: i32) -> Self {
        let start_pos = GridPosition::new(start_x, start_y);
        Self {
            id,
            name: name.to_string(),
            position: start_pos,
            target: start_pos,
            path: VecDeque::new(),
            state: RobotState::Idle,
            paused_until: None,
            last_update: 0,
        }
    }

    pub fn set_target(&mut self, target: GridPosition) {
        self.target = target;
        self.state = RobotState::WaitingForPath;
        self.path.clear();
    }

    pub fn has_reached_target(&self) -> bool {
        self.position == self.target && self.state == RobotState::Completed
    }
}

#[derive(Debug, Clone)]
pub struct CollisionAvoidanceSystem {
    pub robots: HashMap<u32, Robot>,
    pub grid: WarehouseGrid,
    pub current_time: u32,
}

impl CollisionAvoidanceSystem {
    pub fn new(grid: WarehouseGrid) -> Self {
        Self {
            robots: HashMap::new(),
            grid,
            current_time: 0,
        }
    }

    pub fn add_robot(&mut self, robot: Robot) {
        self.robots.insert(robot.id, robot);
    }

    pub fn remove_robot(&mut self, robot_id: u32) -> Option<Robot> {
        self.robots.remove(&robot_id)
    }

    pub fn update(&mut self) -> Result<(), String> {
        self.current_time += 1;
        self.grid.update_temporary_obstacles(self.current_time);
        
        // First pass: update paths for robots that need them
        let mut robots_to_update: Vec<(u32, RobotState)> = Vec::new();
        
        for (robot_id, robot) in &mut self.robots {
            if robot.state == RobotState::WaitingForPath {
                if let Some((path, _cost)) = self.grid.find_path(
                    robot.position,
                    robot.target,
                    self.current_time,
                ) {
                    robot.path = path.into();
                    robot.state = RobotState::Moving;
                    info!("Robot {} calculated path to ({}, {})", 
                          robot.name, robot.target.x, robot.target.y);
                } else {
                    robot.state = RobotState::Error;
                    warn!("Robot {} cannot find path to target", robot.name);
                }
            }
            robots_to_update.push((*robot_id, robot.state.clone()));
        }
        
        // Second pass: move robots
        for (robot_id, _) in &robots_to_update {
            if let Some(robot) = self.robots.get_mut(robot_id) {
                self.update_robot_movement(robot)?;
            }
        }
        
        // Third pass: resolve collisions
        self.resolve_collisions()?;
        
        Ok(())
    }

    fn update_robot_movement(&mut self, robot: &mut Robot) -> Result<(), String> {
        robot.last_update = self.current_time;

        match robot.state {
            RobotState::Paused => {
                if let Some(pause_until) = robot.paused_until {
                    if self.current_time >= pause_until {
                        robot.state = RobotState::Moving;
                        robot.paused_until = None;
                        info!("Robot {} resumed movement", robot.name);
                    }
                }
            }
            RobotState::Moving => {
                if let Some(next_pos) = robot.path.front() {
                    if self.is_position_safe(next_pos, robot.id) {
                        // Move to next position
                        robot.position = *next_pos;
                        robot.path.pop_front();
                        
                        info!("Robot {} moved to ({}, {})", 
                              robot.name, robot.position.x, robot.position.y);
                        
                        // Check if reached target
                        if robot.path.is_empty() {
                            if robot.position == robot.target {
                                robot.state = RobotState::Completed;
                                info!("Robot {} reached target", robot.name);
                            } else {
                                robot.state = RobotState::WaitingForPath;
                            }
                        }
                    } else {
                        // Collision detected - use time-slice mechanism
                        self.handle_collision(robot);
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }

    fn is_position_safe(&self, position: &GridPosition, robot_id: u32) -> bool {
        if !self.grid.is_accessible(position, self.current_time) {
            return false;
        }

        // Check for other robots at this position
        for (other_id, other_robot) in &self.robots {
            if *other_id != robot_id && &other_robot.position == position {
                return false;
            }
            
            // Also check if other robots are moving to this position
            if *other_id != robot_id {
                if let Some(next_pos) = other_robot.path.front() {
                    if next_pos == position && other_robot.state == RobotState::Moving {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn handle_collision(&mut self, robot: &mut Robot) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        if rng.gen_bool(0.6) { // 60% chance to pause this robot
            robot.state = RobotState::Paused;
            robot.paused_until = Some(self.current_time + 2); // Pause for 2 time steps
            
            // Mark the contested position as temporarily occupied
            if let Some(next_pos) = robot.path.front() {
                self.grid.temporary_obstacles.insert(*next_pos, self.current_time + 3);
            }
            
            info!("Robot {} paused due to collision", robot.name);
        }
    }

    fn resolve_collisions(&mut self) -> Result<(), String> {
        let positions: HashMap<u32, GridPosition> = self.robots
            .iter()
            .map(|(id, robot)| (*id, robot.position))
            .collect();

        // Detect position conflicts
        let mut position_counts: HashMap<GridPosition, Vec<u32>> = HashMap::new();
        for (id, pos) in &positions {
            position_counts.entry(*pos).or_default().push(*id);
        }

        // Resolve conflicts - only one robot can occupy a position
        for (pos, robots_at_pos) in position_counts {
            if robots_at_pos.len() > 1 {
                warn!("Collision detected at position ({}, {}) with {} robots", 
                      pos.x, pos.y, robots_at_pos.len());
                
                // Keep first robot at position, move others back or pause them
                for (i, &robot_id) in robots_at_pos.iter().enumerate() {
                    if i > 0 { // Keep first robot, handle others
                        if let Some(robot) = self.robots.get_mut(&robot_id) {
                            // Try to move back to previous position if possible
                            if let Some(prev_pos) = robot.path.back().cloned() {
                                robot.position = prev_pos;
                                robot.state = RobotState::Paused;
                                robot.paused_until = Some(self.current_time + 3);
                                info!("Robot {} moved back due to collision", robot.name);
                            } else {
                                robot.state = RobotState::Paused;
                                robot.paused_until = Some(self.current_time + 5);
                                warn!("Robot {} stuck in collision, pausing", robot.name);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    pub fn get_robot_positions(&self) -> Vec<(u32, GridPosition)> {
        self.robots.iter()
            .map(|(id, robot)| (*id, robot.position))
            .collect()
    }

    pub fn get_robot_status(&self, robot_id: u32) -> Option<String> {
        self.robots.get(&robot_id).map(|robot| {
            format!("Robot {} at ({}, {}) - {:?}", 
                   robot.name, robot.position.x, robot.position.y, robot.state)
        })
    }
}

#[derive(Debug, Clone)]
pub enum Task {
    MoveTo { target: GridPosition },
    PickItem { item_id: String, location: GridPosition },
    PlaceItem { item_id: String, location: GridPosition },
    Wait { duration: u32 },
}

#[derive(Debug, Clone)]
pub struct TaskRobot {
    pub robot: Robot,
    pub current_task: Option<Task>,
    pub task_queue: VecDeque<Task>,
    pub task_state: TaskState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskState {
    Idle,
    ExecutingTask,
    WaitingForTask,
    TaskFailed,
    TaskCompleted,
}

pub struct TaskManager {
    pub task_robots: HashMap<u32, TaskRobot>,
    pub collision_system: CollisionAvoidanceSystem,
}

impl TaskManager {
    pub fn new(grid: WarehouseGrid) -> Self {
        Self {
            task_robots: HashMap::new(),
            collision_system: CollisionAvoidanceSystem::new(grid),
        }
    }

    pub fn add_robot(&mut self, robot: Robot) {
        let task_robot = TaskRobot {
            robot: robot.clone(),
            current_task: None,
            task_queue: VecDeque::new(),
            task_state: TaskState::Idle,
        };
        self.collision_system.add_robot(robot);
        self.task_robots.insert(task_robot.robot.id, task_robot);
    }

    pub fn update(&mut self) -> Result<(), String> {
        // Update collision system first
        self.collision_system.update()?;
        
        // Update task states for each robot
        let robot_ids: Vec<u32> = self.task_robots.keys().cloned().collect();
        
        for robot_id in robot_ids {
            if let Some(task_robot) = self.task_robots.get_mut(&robot_id) {
                // Sync robot state from collision system
                if let Some(updated_robot) = self.collision_system.robots.get(&robot_id) {
                    task_robot.robot = updated_robot.clone();
                }
                
                Self::update_task_robot(task_robot, &mut self.collision_system);
            }
        }
        
        Ok(())
    }

    fn update_task_robot(task_robot: &mut TaskRobot, collision_system: &mut CollisionAvoidanceSystem) {
        match task_robot.task_state {
            TaskState::Idle | TaskState::TaskCompleted | TaskState::TaskFailed => {
                if let Some(task) = task_robot.task_queue.pop_front() {
                    task_robot.current_task = Some(task.clone());
                    task_robot.task_state = TaskState::ExecutingTask;
                    Self::start_task(task_robot, collision_system);
                    info!("Robot {} started new task: {:?}", task_robot.robot.name, task);
                } else {
                    task_robot.task_state = TaskState::WaitingForTask;
                }
            }
            TaskState::ExecutingTask => {
                Self::execute_task(task_robot);
            }
            _ => {}
        }
    }

    fn start_task(task_robot: &mut TaskRobot, collision_system: &mut CollisionAvoidanceSystem) {
        if let Some(ref task) = task_robot.current_task {
            match task {
                Task::MoveTo { target } => {
                    if let Some(robot) = collision_system.robots.get_mut(&task_robot.robot.id) {
                        robot.set_target(*target);
                    }
                }
                Task::PickItem { item_id, location } => {
                    info!("Robot {} picking item {} at ({}, {})", 
                          task_robot.robot.name, item_id, location.x, location.y);
                    // Simulate pick operation
                    task_robot.task_state = TaskState::TaskCompleted;
                }
                Task::PlaceItem { item_id, location } => {
                    info!("Robot {} placing item {} at ({}, {})", 
                          task_robot.robot.name, item_id, location.x, location.y);
                    // Simulate place operation
                    task_robot.task_state = TaskState::TaskCompleted;
                }
                Task::Wait { duration } => {
                    if let Some(robot) = collision_system.robots.get_mut(&task_robot.robot.id) {
                        robot.state = RobotState::Paused;
                        robot.paused_until = Some(collision_system.current_time + duration);
                    }
                    task_robot.task_state = TaskState::TaskCompleted;
                }
            }
        }
    }

    fn execute_task(task_robot: &mut TaskRobot) {
        if let Some(ref task) = task_robot.current_task {
            match task {
                Task::MoveTo { target: _ } => {
                    if task_robot.robot.state == RobotState::Completed {
                        task_robot.task_state = TaskState::TaskCompleted;
                        info!("Robot {} completed move task", task_robot.robot.name);
                    } else if task_robot.robot.state == RobotState::Error {
                        task_robot.task_state = TaskState::TaskFailed;
                        warn!("Robot {} failed move task", task_robot.robot.name);
                    }
                }
                Task::Wait { duration: _ } => {
                    if task_robot.robot.state != RobotState::Paused {
                        task_robot.task_state = TaskState::TaskCompleted;
                    }
                }
                _ => {
                    // Other task types handle completion in start_task
                }
            }
        }
    }

    pub fn assign_task(&mut self, robot_id: u32, task: Task) -> Result<(), String> {
        if let Some(task_robot) = self.task_robots.get_mut(&robot_id) {
            task_robot.task_queue.push_back(task);
            Ok(())
        } else {
            Err(format!("Robot {} not found", robot_id))
        }
    }

    pub fn get_robot_status(&self, robot_id: u32) -> Option<String> {
        self.collision_system.get_robot_status(robot_id)
    }

    pub fn get_all_robot_status(&self) -> Vec<String> {
        self.task_robots.keys()
            .filter_map(|id| self.get_robot_status(*id))
            .collect()
    }
}

// 异步主程序
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt::init();
    info!("Starting Warehouse Robot System");

    // 创建仓库网格 (20x20)
    let mut grid = WarehouseGrid::new(20, 20);
    
    // 添加障碍物
    for x in 5..=15 {
        grid.add_obstacle(x, 10); // 水平障碍物
    }
    for y in 5..=15 {
        grid.add_obstacle(10, y); // 垂直障碍物
    }
    
    // 添加一些随机障碍物
    grid.add_obstacle(3, 3);
    grid.add_obstacle(7, 7);
    grid.add_obstacle(15, 5);
    grid.add_obstacle(12, 15);

    // 创建任务管理器
    let task_manager = Arc::new(RwLock::new(TaskManager::new(grid)));

    // 添加机器人
    {
        let mut tm = task_manager.write().await;
        tm.add_robot(Robot::new(1, "Robot-1", 0, 0));
        tm.add_robot(Robot::new(2, "Robot-2", 0, 1));
        tm.add_robot(Robot::new(3, "Robot-3", 1, 0));
        info!("Added 3 robots to the system");
    }

    // 分配任务
    {
        let mut tm = task_manager.write().await;
        
        // Robot-1 移动到对面角落
        tm.assign_task(1, Task::MoveTo { target: GridPosition::new(19, 19) })?;
        
        // Robot-2 移动到中间位置
        tm.assign_task(2, Task::MoveTo { target: GridPosition::new(8, 8) })?;
        
        // Robot-3 执行一系列任务
        tm.assign_task(3, Task::MoveTo { target: GridPosition::new(15, 15) })?;
        tm.assign_task(3, Task::PickItem { 
            item_id: "ITEM-001".to_string(), 
            location: GridPosition::new(15, 15) 
        })?;
        tm.assign_task(3, Task::MoveTo { target: GridPosition::new(0, 0) })?;
        tm.assign_task(3, Task::PlaceItem { 
            item_id: "ITEM-001".to_string(), 
            location: GridPosition::new(0, 0) 
        })?;
        
        info!("Assigned tasks to all robots");
    }

    // 主循环 - 模拟系统运行
    info!("Starting simulation loop");
    for step in 0..100 {
        println!("\n=== Step {} ===", step);
        
        {
            let mut tm = task_manager.write().await;
            if let Err(e) = tm.update() {
                eprintln!("Error updating task manager: {}", e);
                break;
            }
            
            // 打印所有机器人状态
            for status in tm.get_all_robot_status() {
                println!("{}", status);
            }
        }
        
        // 检查是否所有任务完成
        {
            let tm = task_manager.read().await;
            let all_idle = tm.task_robots.values()
                .all(|tr| tr.task_queue.is_empty() && tr.current_task.is_none());
            
            if all_idle {
                info!("All tasks completed!");
                break;
            }
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    info!("Simulation completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathfinding_basic() {
        let mut grid = WarehouseGrid::new(10, 10);
        grid.add_obstacle(5, 5);
        
        let start = GridPosition::new(0, 0);
        let goal = GridPosition::new(9, 9);
        
        let path = grid.find_path(start, goal, 0);
        assert!(path.is_some());
        
        let (path_positions, cost) = path.unwrap();
        assert!(!path_positions.is_empty());
        assert!(cost > 0);
        println!("Path found with cost {}: {:?}", cost, path_positions);
    }

    #[test]
    fn test_robot_movement() {
        let grid = WarehouseGrid::new(10, 10);
        let mut robot = Robot::new(1, "TestRobot", 0, 0);
        
        robot.set_target(GridPosition::new(5, 5));
        assert_eq!(robot.state, RobotState::WaitingForPath);
        assert_eq!(robot.target, GridPosition::new(5, 5));
    }

    #[tokio::test]
    async fn test_task_assignment() {
        let grid = WarehouseGrid::new(10, 10);
        let mut task_manager = TaskManager::new(grid);
        
        task_manager.add_robot(Robot::new(1, "TestRobot", 0, 0));
        
        let result = task_manager.assign_task(1, Task::MoveTo { 
            target: GridPosition::new(5, 5) 
        });
        
        assert!(result.is_ok());
    }
}
