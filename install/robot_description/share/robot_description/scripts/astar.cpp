#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <vector>
#include <algorithm>

struct Cell {
    int xi, yi;
    bool operator==(const Cell& other) const {
        return xi == other.xi && yi == other.yi;
    }
};

struct CellHasher {
    std::size_t operator()(const Cell& c) const {
        return std::hash<int>()(c.xi) ^ (std::hash<int>()(c.yi) << 1);
    }
};

class AStar : public rclcpp::Node {
public:
    AStar() : Node("astar_node") {
        auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local();
        auto default_qos = rclcpp::QoS(rclcpp::KeepLast(10));

        map_subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", map_qos, std::bind(&AStar::map_cb, this, std::placeholders::_1));
        
        pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/amcl_pose", default_qos, std::bind(&AStar::pose_cb, this, std::placeholders::_1));
            
        goal_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", default_qos, std::bind(&AStar::goal_cb, this, std::placeholders::_1));
        
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/global_path", default_qos);
        
        RCLCPP_INFO(this->get_logger(), "A* Planner Node Initialized. Waiting for map, pose, and goal...");
    }

private:
    struct PointNode {
        int x, y;
        double g, f;
        PointNode* prev;

        PointNode(int x, int y) : x(x), y(y), 
                                   g(std::numeric_limits<double>::infinity()),
                                   f(std::numeric_limits<double>::infinity()), 
                                   prev(nullptr) {}
    };

    void map_cb(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        grid_ = msg;
        ready_map_ = true;
        RCLCPP_INFO(this->get_logger(), "Map received in A* Node.");
    }

    void pose_cb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        robot_pose_ = msg->pose.pose;
        ready_pose_ = true;
    }

    void goal_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        if (!ready_map_ || !ready_pose_) {
            RCLCPP_WARN(this->get_logger(), "Cannot plan: Map or AMCL Pose not ready!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Goal received! Computing A* path...");
        nav_msgs::msg::Path planned_path = compute(robot_pose_, msg->pose);
        path_publisher_->publish(planned_path);
    }

    bool check_bounds(int x, int y) {
        if (!ready_map_ || x < 0 || x >= (int)grid_->info.width || y < 0 || y >= (int)grid_->info.height)
            return false;
        int val = grid_->data[y * grid_->info.width + x];
        return (val >= 0 && val < 50); 
    }

    double get_h(int x1, int y1, int x2, int y2) {
        return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
    }

    nav_msgs::msg::Path compute(const geometry_msgs::msg::Pose& start, const geometry_msgs::msg::Pose& goal) {
        nav_msgs::msg::Path final_path;
        final_path.header.frame_id = "map";
        final_path.header.stamp = this->now();

        int sx = (start.position.x - grid_->info.origin.position.x) / grid_->info.resolution;
        int sy = (start.position.y - grid_->info.origin.position.y) / grid_->info.resolution;
        int gx = (goal.position.x - grid_->info.origin.position.x) / grid_->info.resolution;
        int gy = (goal.position.y - grid_->info.origin.position.y) / grid_->info.resolution;

        typedef std::pair<double, PointNode*> QueueElement;
        std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> open_list;
        
        std::unordered_map<Cell, PointNode*, CellHasher> nodes_cache;

        PointNode* start_node = new PointNode(sx, sy);
        start_node->g = 0.0;
        start_node->f = get_h(sx, sy, gx, gy);
        
        open_list.push({start_node->f, start_node});
        nodes_cache[{sx, sy}] = start_node;

        PointNode* target_reached = nullptr;

        while (!open_list.empty()) {
            auto current_elem = open_list.top();
            open_list.pop();
            
            PointNode* current = current_elem.second;

            if (current->f < current_elem.first) continue;

            if (current->x == gx && current->y == gy) {
                target_reached = current;
                break;
            }

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if (i == 0 && j == 0) continue;

                    int tx = current->x + i;
                    int ty = current->y + j;

                    if (check_bounds(tx, ty)) {
                        double step = (i == 0 || j == 0) ? 1.0 : 1.414; 
                        double new_g = current->g + step;

                        Cell c = {tx, ty};
                        if (nodes_cache.find(c) == nodes_cache.end() || new_g < nodes_cache[c]->g) {
                            PointNode* neighbor = (nodes_cache.find(c) == nodes_cache.end()) ? new PointNode(tx, ty) : nodes_cache[c];
                            
                            neighbor->prev = current;
                            neighbor->g = new_g;
                            neighbor->f = new_g + get_h(tx, ty, gx, gy);
                            
                            nodes_cache[c] = neighbor;
                            open_list.push({neighbor->f, neighbor});
                        }
                    }
                }
            }
        }

        if (target_reached) {
            PointNode* curr = target_reached;
            while (curr) {
                geometry_msgs::msg::PoseStamped ps;
                ps.header.frame_id = "map";
                ps.pose.position.x = (curr->x * grid_->info.resolution) + grid_->info.origin.position.x + (grid_->info.resolution / 2.0);
                ps.pose.position.y = (curr->y * grid_->info.resolution) + grid_->info.origin.position.y + (grid_->info.resolution / 2.0);
                ps.pose.orientation.w = 1.0;
                final_path.poses.push_back(ps);
                curr = curr->prev;
            }
            std::reverse(final_path.poses.begin(), final_path.poses.end());
            RCLCPP_INFO(this->get_logger(), "Path found with %zu poses!", final_path.poses.size());
        } else {
            RCLCPP_WARN(this->get_logger(), "No valid path found to the goal.");
        }

        for (auto& n : nodes_cache) delete n.second;
        return final_path;
    }

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_subscription_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;

    nav_msgs::msg::OccupancyGrid::SharedPtr grid_;
    geometry_msgs::msg::Pose robot_pose_;
    bool ready_map_ = false;
    bool ready_pose_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AStar>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}