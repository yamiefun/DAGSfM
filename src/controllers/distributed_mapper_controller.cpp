#include "controllers/distributed_mapper_controller.h"
#include "controllers/sfm_aligner.h"
#include "base/database.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/reconstruction_io.h"

#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <ceres/rotation.h>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <dirent.h>
#include <set>

using namespace colmap;

namespace GraphSfM {

bool DistributedMapperController::Options::Check() const
{
    CHECK_OPTION_GT(num_workers, -1);
    return true;
}


DistributedMapperController::DistributedMapperController(
    const Options& options,
    const ImageClustering::Options& clustering_options,
    const IncrementalMapperOptions& mapper_options,
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      clustering_options_(clustering_options),
      mapper_options_(mapper_options),
      reconstruction_manager_(reconstruction_manager)
{
    CHECK(options.Check());
}


void DistributedMapperController::Run()
{
    //////////////////////////////////////////////////////////////////
    // 1. Partitioning the images into the given number of clusters //
    //////////////////////////////////////////////////////////////////
    PrintHeading1("Partitioning the Scene...");

    std::vector<std::pair<image_t, image_t>> image_pairs;
    std::vector<int> num_inliers;
    std::unordered_map<image_t, std::string> image_id_to_name;
    std::vector<image_t> image_ids;
    LoadData(image_pairs, num_inliers, image_id_to_name, image_ids);

    // For each VIO file, create a vector record ts -> VIO mapping.
    // There may be multiple files, so it's a vector of vector.
    std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>> ts_to_VIO;

    // Store each VIO file's starting ts and ending ts.
    std::vector<std::pair<std::string, std::string>> VIO_summary;
    LoadVIO(ts_to_VIO, VIO_summary);

    // Find VIO threshold (height and drift thresh)
    std::vector<GraphSfM::VIOThresh> vio_thresh;
    CalculateVIOThresh(vio_thresh, ts_to_VIO);
    LOG(INFO) << "XDDDDDDDDheight thresh: " << vio_thresh[0].height_thresh << std::endl;

    // Mapping from image_id to image information, including VIO info.
    std::unordered_map<image_t, GraphSfM::ImageInfo> image_information;
    MatchImageWithCoordinate(
        image_information, image_id_to_name, ts_to_VIO, VIO_summary);
    // for (const auto & image_info : image_information){
    //     LOG(INFO) << image_info.first << std::endl;
    // }

    ImageCluster image_cluster;
    std::vector<ImageCluster> inter_clusters;
    if (!VIO_summary.empty()){
        // create viewing graph with VIO
        LOG(INFO) << "Found VIO file.\n";
        CreateImageViewGraph(
            image_information, image_pairs, num_inliers, image_ids, image_cluster, vio_thresh);
        LOG(INFO) << "FINISH HERE\n";
        return;
        // clustering
        inter_clusters = ClusteringScenesWithViewGraph(image_cluster);
    }
    else{
        LOG(INFO) << "No VIO file found...\n";
        inter_clusters = ClusteringScenes(image_pairs, num_inliers, image_ids);
    }
    

    // std::unique_ptr<ImageClustering> image_clustering_;
    

    

    // ////////////////////////////////////////////////////////////////
    // // 2. Reconstruct all clusters in parallel/distributed manner //
    // ////////////////////////////////////////////////////////////////
    PrintHeading1("Reconstucting Clusters...");

    std::unordered_map<const ImageCluster*, ReconstructionManager> reconstruction_managers;
    std::vector<Reconstruction*> reconstructions;
    ReconstructPartitions(image_id_to_name, inter_clusters, reconstruction_managers, reconstructions);    

    // ////////////////////////////////////////////////
    // // 3. Merge clusters ///////////////////////////
    // ////////////////////////////////////////////////
    PrintHeading1("Merging Clusters...");

    // Determine the number of workers and threads per worker
    const int kMaxNumThreads = -1;
    const int num_eff_threads = GetEffectiveNumThreads(kMaxNumThreads);
    MergeClusters(inter_clusters, 
                  reconstructions, 
                  reconstruction_managers,
                  num_eff_threads);

    std::cout << std::endl;
    GetTimer().PrintMinutes();
}


BundleAdjustmentOptions DistributedMapperController::GlobalBundleAdjustment() const 
{
    BundleAdjustmentOptions options;
    options.solver_options.function_tolerance = 0.0;
    options.solver_options.gradient_tolerance = 1.0;
    options.solver_options.parameter_tolerance = 0.0;
    options.solver_options.max_num_iterations = 50;
    options.solver_options.max_linear_solver_iterations = 100;
    options.solver_options.minimizer_progress_to_stdout = true;
    options.solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
    options.solver_options.num_linear_solver_threads = GetEffectiveNumThreads(-1);
#endif  // CERES_VERSION_MAJOR
    options.print_summary = true;
    options.refine_focal_length = true;
    options.refine_principal_point = false;
    options.refine_extra_params = true;
    options.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
    return options;
}


bool DistributedMapperController::IsPartialReconsExist(std::vector<Reconstruction*>& recons) const
{
    const std::vector<std::string> dirs = colmap::GetRecursiveDirList(options_.output_path);
    recons.reserve(dirs.size());
    for (auto path : dirs) {
        Reconstruction* recon = new Reconstruction();
        if (ExistsFile(JoinPaths(path, "cameras.bin")) &&
            ExistsFile(JoinPaths(path, "images.bin")) &&
            ExistsFile(JoinPaths(path, "points3D.bin"))) {
            recon->ReadBinary(path);
        } else if (ExistsFile(JoinPaths(path, "cameras.txt")) &&
                   ExistsFile(JoinPaths(path, "images.txt")) &&
                   ExistsFile(JoinPaths(path, "points3D.txt"))) {
            recon->ReadText(path);
        } else {
            LOG(WARNING) << "cameras, images, points3D files do not exist at " << path;
            continue;
        }
        recons.push_back(recon);
    }

    if (recons.empty()) return false;
    return true;
}


void DistributedMapperController::LoadData(
    std::vector<std::pair<image_t, image_t>>& image_pairs,
    std::vector<int>& num_inliers,
    std::unordered_map<image_t, std::string>& image_id_to_name,
    std::vector<image_t>& image_ids)
{ 
    // Loading database
    Database database(options_.database_path);

    // Reading all images
    LOG(INFO) << "Reading images...";
    const auto images = database.ReadAllImages();
    for (const auto& image : images) {
        image_id_to_name.emplace(image.ImageId(), image.Name());
        image_ids.push_back(image.ImageId());
    }

    // Reading scene graph
    LOG(INFO) << "Reading scene graph...";
    CHECK_EQ(image_pairs.size(), num_inliers.size());
    database.ReadTwoViewGeometryNumInliers(&image_pairs, &num_inliers);
}


void DistributedMapperController::LoadVIO(
    std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>>& ts_to_VIO,
    std::vector<std::pair<std::string, std::string>>& VIO_summary)
{
    LOG(INFO) << "Parsing VIO file in " << options_.VIO_folder_path << "...";
    namespace b_fs = boost::filesystem;
    DIR *dr;
    struct dirent *d;
    dr = opendir(options_.VIO_folder_path.c_str());
    b_fs::path dir (options_.VIO_folder_path);

    // get all VIO files
    std::vector<std::string> VIO_files;
    if (dr != NULL){
        for (d = readdir (dr); d != NULL; d = readdir(dr)){
            b_fs::path file (d->d_name);
            if (file.string() == "." || file.string() == "..")
                continue;
            b_fs::path full_path = dir / file;
            // LOG(INFO) << full_path.string() << std::endl;
            VIO_files.push_back(full_path.string());
        }
        closedir(dr);
    }

    // loop and read each VIO file
    for (const auto & vio_file_path : VIO_files){
        std::vector<std::pair<std::string, GraphSfM::Coordinate>> ts_VIO_pairs;
        std::fstream vio_file;
        vio_file.open(vio_file_path);
        std::string delimiter = ",";
        size_t pos = 0;
        std::string substring;
        std::string line;
        while (getline(vio_file, line)){
            uint parse_idx = 0;
            std::string ts;
            GraphSfM::Coordinate coordinate;
            // parse first 4 element in each line: ts, x, y, z
            while ((pos = line.find(delimiter)) != std::string::npos && parse_idx < 4){
                substring = line.substr(0, pos);
                line.erase(0, pos + delimiter.length());
                if (parse_idx == 0)
                    ts = substring;
                else if (parse_idx == 1)
                    coordinate.x = std::stod(substring);
                else if (parse_idx == 2)
                    coordinate.z = std::stod(substring);
                else if (parse_idx == 3)
                    coordinate.y = std::stod(substring);
                parse_idx++;
            }
            ts_VIO_pairs.push_back(std::make_pair(ts, coordinate));
        }
        ts_to_VIO.push_back(ts_VIO_pairs);
        vio_file.close();
    }

    // store each VIO ts range start and end
    for (const auto & ts_VIO_pairs : ts_to_VIO){
        VIO_summary.push_back(
            std::make_pair(ts_VIO_pairs.front().first, ts_VIO_pairs.back().first)
        );
    }
}


double DistributedMapperController::square_distance_3d(
    GraphSfM::Coordinate& coord1, GraphSfM::Coordinate& coord2
)
{
    return pow(coord1.x - coord2.x, 2) +
           pow(coord1.y - coord2.y, 2) +
           pow(coord1.z - coord2.z, 2);
}


void DistributedMapperController::CalculateVIOThresh(
    std::vector<GraphSfM::VIOThresh>& vio_thresh,
    std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>>& ts_to_VIO)
{
    auto compare_height = 
        [](std::pair<std::string, GraphSfM::Coordinate> vio_info1,
           std::pair<std::string, GraphSfM::Coordinate> vio_info2) -> bool
        {return vio_info1.second.y < vio_info2.second.y;};
    for (auto & ts_VIO_pairs : ts_to_VIO){
        // Height threshold:
        // Find highest and lowest VIO info, divided by a magic number 8.
        double highest = std::max_element(
            ts_VIO_pairs.begin(), ts_VIO_pairs.end(), compare_height)->second.y;
        double lowest = std::min_element(
            ts_VIO_pairs.begin(), ts_VIO_pairs.end(), compare_height)->second.y;
        double height_thresh = (highest - lowest) / 8;

        // Drift threshold:
        // Find 3d distance between every adjacent VIO coordinates, divided by number of VIO info,
        // multiplied by a magic number 20.
        double drift_sum = 0.0;
        double drift_thresh;
        for (std::size_t i = 1; i < ts_VIO_pairs.size(); ++i){
            drift_sum += square_distance_3d(
                ts_VIO_pairs[i].second, ts_VIO_pairs[i-1].second);
        }
        drift_thresh = 40 * (drift_sum / ts_VIO_pairs.size());
        height_thresh = 0.15;
        GraphSfM::VIOThresh thresh{height_thresh, drift_thresh};
        vio_thresh.push_back(thresh);
    }
}


std::string DistributedMapperController::ParseImageNameFromPath(std::string path){
    std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
    std::string::size_type const p(base_filename.find_last_of('.'));
    std::string file_without_extension = base_filename.substr(0, p);
    return file_without_extension;
}


bool DistributedMapperController::CHECK_TS_LE(std::string ts1, std::string ts2){
    return std::strcmp(ts1.c_str(), ts2.c_str()) <= 0;
}


GraphSfM::VIOInfo DistributedMapperController::CheckImageWithVIO(
    std::vector<std::pair<std::string, std::string>>& VIO_summary,
    std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>>& ts_to_VIO,
    std::string& target)
{
    GraphSfM::VIOInfo VIO_info;
    VIO_info.has_VIO = false;

    // not legal timestamp format
    if (target.length() != 19)
        return VIO_info;
    for (uint i = 0; i < 19; i++)
        if (std::isdigit(target[i]) == 0)
            return VIO_info;

    // not in VIO data timestamp range
    int belong_ts_group = -1;
    for (std::size_t i = 0; i < VIO_summary.size(); i++){
        std::string start = VIO_summary[i].first;
        std::string end = VIO_summary[i].second;
        if (CHECK_TS_LE(start, target) && CHECK_TS_LE(target, end)){
            belong_ts_group = i;
            break;
        }
    }
    if (belong_ts_group == -1)
        return VIO_info;
    
    // legal timestamp, find cooresponding VIO coordinate
    VIO_info.has_VIO = true;

    auto compare_timestamp = 
        [](std::pair<std::string, Coordinate> comparison, std::string target) -> bool
        { return std::strcmp(comparison.first.c_str(), target.c_str()) < 0; };

    // find the lower bound timestamp and match VIO information
    // to improve, find "closest timestamp" rather than "lower bound"
    auto it = std::lower_bound(
        ts_to_VIO[belong_ts_group].begin(), ts_to_VIO[belong_ts_group].end(),
        target, compare_timestamp);
    VIO_info.coordinate = it->second;
    VIO_info.VIO_group_id = belong_ts_group;
    return VIO_info;
}


void DistributedMapperController::MatchImageWithCoordinate(
    std::unordered_map<image_t, GraphSfM::ImageInfo>& image_information,
    std::unordered_map<image_t, std::string>& image_id_to_name,
    std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>>& ts_to_VIO,
    std::vector<std::pair<std::string, std::string>>& VIO_summary)
{
    for (const auto& image : image_id_to_name){
        std::string image_name_with_path = image.second;
        std::string image_name = ParseImageNameFromPath(image_name_with_path);
        GraphSfM::VIOInfo VIO_info = CheckImageWithVIO(
            VIO_summary, ts_to_VIO, image_name);
        if (VIO_info.has_VIO){
            GraphSfM::ImageInfo image_info;
            image_info.image_name = image_name;
            image_info.VIO_info = VIO_info;
            image_information.emplace(image.first, image_info);
        }
    }
}


bool DistributedMapperController::CHECK_STR_LT(std::string& ts1, std::string& ts2){
    for (uint i = 0; i < ts1.length(); ++i){
        if (ts1[i] < ts2[i])
            return true;
        else if (ts1[i] > ts2[i])
            return false;
    }
    return false;
}


int DistributedMapperController::ts_diff(std::string ts1, std::string ts2){
    // Calculate two timestamp diff.
    // Assume both timestamp are legal with same length.

    if (CHECK_STR_LT(ts1, ts2))
        swap(ts1, ts2);
    std::string subtract = "";
    int carry = 0;
    for (int i = ts1.length()-1; i >= 0; --i){
        int sub = ((ts1[i] - '0') - (ts2[i] -  '0') - carry);
        if (sub < 0){
            sub += 10;
            carry = 1;
        }
        else{
            carry = 0;
        }
        subtract.push_back(sub + '0');
    }
    reverse(subtract.begin(), subtract.end());
    std::string second = subtract.substr(0, 10);
    second.erase(0, std::min(second.find_first_not_of('0'), second.size()-1));
    return std::stoi(second);
}


void DistributedMapperController::CreateImageViewGraph(
    std::unordered_map<image_t, GraphSfM::ImageInfo>& image_information,
    std::vector<std::pair<image_t, image_t>>& image_pairs,
    std::vector<int>& num_inliers,
    std::vector<image_t>& image_ids,
    ImageCluster& image_cluster,
    std::vector<GraphSfM::VIOThresh>& vio_thresh)
{
    LOG(INFO) << "Creating view graph with VIO..." << std::endl;
    // double coor_thresh; 
    int ts_thresh = 30;

    // collect all edges with weight
    std::unordered_map<ViewIdPair, int> edges;
    for (uint i = 0; i < image_pairs.size(); ++i) {
        const ViewIdPair view_pair(image_pairs[i].first, image_pairs[i].second);
        edges[view_pair] = num_inliers[i];
    }

    // collect all image ids which has vio info
    std::set<image_t> images_with_vio;
    for (const auto & image : image_information){
        if (image.second.VIO_info.has_VIO)
            images_with_vio.insert(image.first);
    }

    // memo of images which are added into graph already
    std::set<image_t> image_added;

    // 1. Consider edges which both vertices have VIO info
    LOG(INFO) << "Building view graph's arterial..." << std::endl;
    for (const auto & image1 : images_with_vio){
        for (const auto & image2 : images_with_vio){
            // redundant, assure id1 < id2
            if (image1 >= image2)
                continue;

            // Same image or two images not in same ts range
            if (image_information[image1].VIO_info.VIO_group_id !=
                image_information[image2].VIO_info.VIO_group_id)
                continue;

            const ViewIdPair view_pair(image1, image2);
            if (edges.find(view_pair) == edges.end())
                continue;
            // Check ts diff between two image
            if (ts_diff(image_information[image1].image_name,
                        image_information[image2].image_name) <= ts_thresh){
                image_cluster.edges[view_pair] = edges[view_pair];
                if (image_added.find(image1) == image_added.end()){
                    image_cluster.image_ids.push_back(image1);
                    image_added.insert(image1);
                }
                if (image_added.find(image2) == image_added.end()){
                    image_cluster.image_ids.push_back(image2);
                    image_added.insert(image2);
                }
            }

            // Consider drift thresh and height thresh
            GraphSfM::Coordinate coord1 = image_information[image1].VIO_info.coordinate;
            GraphSfM::Coordinate coord2 = image_information[image2].VIO_info.coordinate;
            uint group_id = image_information[image1].VIO_info.VIO_group_id;
            // if (square_distance_3d(coord1, coord2) <= vio_thresh[group_id].drift_thresh ||
            //         abs(coord1.y - coord2.y) <= vio_thresh[group_id].height_thresh){
            // if (square_distance_3d(coord1, coord2) <= vio_thresh[group_id].drift_thresh){
            // if (abs(coord1.y - coord2.y) < vio_thresh[group_id].height_thresh){
            //     LOG(INFO) << "height thresh " << vio_thresh[group_id].height_thresh << std::endl;
            //     image_cluster.edges[view_pair] = edges[view_pair];
            //     if (image_added.find(image1) == image_added.end()){
            //         image_cluster.image_ids.push_back(image1);
            //         image_added.insert(image1);
            //     }
            //     if (image_added.find(image2) == image_added.end()){
            //         image_cluster.image_ids.push_back(image2);
            //         image_added.insert(image2);
            //     }
            // }
        }
    }
    // 2. Consider rest of edges
    int image_remain = image_ids.size() - image_added.size();
    LOG(INFO) << "Add rest images into view graph, image nums: " << image_remain << std::endl;
    double weight_vio = 1.0;
    double weight_no_vio = 0.8;
    while (image_remain > 0){
        LOG(INFO) << "remain image num: " << image_remain << std::endl;
        double best_score = 0.0;
        image_t best_id;
        for (auto & image_id : image_ids){
            // check if image is already in graph
            if (image_added.find(image_id) != image_added.end())
                continue;
            double cur_score;
            cur_score = CalculateScore(
                image_id, image_added, edges, images_with_vio, weight_vio, weight_no_vio);
            if (cur_score > best_score){
                best_score = cur_score;
                best_id = image_id;
            }
        }
        LOG(INFO) << "Add image score: " << best_score << std::endl;
        // add all edges from best_id image to images that are in graph already
        // weight the edges depends on the reference image has VIO or not
        for (auto & image_id : image_added){
            if (best_id == image_id)
                continue;
            ViewIdPair view_pair;
            view_pair = (best_id < image_id) ?
                std::make_pair(best_id, image_id) :
                std::make_pair(image_id, best_id);

            // const ViewIdPair view_pair(best_id, image_id);
            if (edges.find(view_pair) != edges.end()){
                if (images_with_vio.find(image_id) != images_with_vio.end()){
                    image_cluster.edges[view_pair] = edges[view_pair] * weight_vio;
                }
                else{
                    image_cluster.edges[view_pair] = edges[view_pair] * weight_no_vio;
                }
            }
        }
        image_remain--;

        // add image to graph
        image_cluster.image_ids.push_back(best_id);
        image_added.insert(best_id);
    }

    // 3. Delete false positive feature matches in database
    LOG(INFO) << "Delete false feature matches in database..." << std::endl;
    Database database(options_.database_path);
    for (uint i = 0; i < image_pairs.size(); ++i){
        image_t image_id1 = image_pairs[i].first;
        image_t image_id2 = image_pairs[i].second;
        const ViewIdPair view_pair(image_id1, image_id2);
        if (image_cluster.edges.find(view_pair) == image_cluster.edges.end()){
            database.DeleteInlierMatches(image_id1, image_id2);
            LOG(INFO) << "Delete inlier match: " << image_id1 << " " << image_id2 << std::endl;
        }
    }

    // Check graph completness
    // LOG(INFO) << "Number of images in graph: " << image_cluster.image_ids.size() << std::endl;
    // LOG(INFO) << "Number of images (total): " << image_ids.size() << std::endl;
}


double DistributedMapperController::CalculateScore(
    image_t& image_id,
    std::set<image_t>& image_added,
    std::unordered_map<ViewIdPair, int>& edges,
    std::set<image_t>& images_with_vio,
    double& weight_vio, double& weight_no_vio)
{
    double best_score = 0.0;
    for (auto & ref_image : image_added){
        // find all reference images
        std::vector<image_t> ref_images;
        ref_images.push_back(ref_image);
        for (auto & ref_neighbor : image_added){
            if (ref_neighbor == ref_image)
                continue;
            ViewIdPair view_pair;
            view_pair = (ref_image < ref_neighbor) ?
                std::make_pair(ref_image, ref_neighbor) :
                std::make_pair(ref_neighbor, ref_image);
            if (edges.find(view_pair) != edges.end()){
                ref_images.push_back(ref_neighbor);
            }
        }

        // mean weighted sum of all reference images with target image
        double cur_score = 0.0;
        uint edge_count = 0;
        for (auto & ref : ref_images){
            ViewIdPair view_pair;
            view_pair = (image_id < ref) ?
                std::make_pair(image_id, ref) :
                std::make_pair(ref, image_id);
            if (edges.find(view_pair) != edges.end()){
                edge_count += 1;
                if (images_with_vio.find(ref) != images_with_vio.end())
                    cur_score += weight_vio * edges[view_pair];
                else
                    cur_score += weight_no_vio * edges[view_pair];
            }
        }
        cur_score = cur_score / edge_count;
        best_score = std::max(best_score, cur_score);
    }
    return best_score;
}


std::vector<ImageCluster> DistributedMapperController::ClusteringScenes(
    const std::vector<std::pair<image_t, image_t>>& image_pairs,
    const std::vector<int>& num_inliers,
    const std::vector<image_t>& image_ids)
{
    // Clustering images
    // create whole viewing graph first
    ImageCluster image_cluster;
    image_cluster.image_ids = image_ids;
    for (uint i = 0; i < image_pairs.size(); i++) {
        const ViewIdPair view_pair(image_pairs[i].first, image_pairs[i].second);
        // VIO trim here
        image_cluster.edges[view_pair] = num_inliers[i];
    }

    // create `image_clustering_` for clustering
    image_clustering_ = std::unique_ptr<ImageClustering>(
                        new ImageClustering(clustering_options_, image_cluster));
    image_clustering_->Cut();
    image_clustering_->Expand();
    // image_clustering_.CutAndExpand();
    image_clustering_->OutputClusteringSummary();
    
    // obtain a vector of clusters
    std::vector<ImageCluster> inter_clusters = image_clustering_->GetInterClusters();
    for (auto cluster : inter_clusters) {
        cluster.ShowInfo();
    }

    return inter_clusters;
}


std::vector<ImageCluster> DistributedMapperController::ClusteringScenesWithViewGraph(
    ImageCluster& image_cluster)
{
    image_clustering_ = std::unique_ptr<ImageClustering>(
                        new ImageClustering(clustering_options_, image_cluster));
    image_clustering_->Cut();
    image_clustering_->Expand();
    // image_clustering_.CutAndExpand();
    image_clustering_->OutputClusteringSummary();
    
    // obtain a vector of clusters
    std::vector<ImageCluster> inter_clusters = image_clustering_->GetInterClusters();
    for (auto cluster : inter_clusters) {
        cluster.ShowInfo();
    }

    return inter_clusters;
}


void DistributedMapperController::ReconstructPartitions(
    const std::unordered_map<image_t, std::string>& image_id_to_name,
    std::vector<ImageCluster>& inter_clusters,
    std::unordered_map<const ImageCluster*, ReconstructionManager>& reconstruction_managers,
    std::vector<Reconstruction*>& reconstructions)
{
    // Determine the number of workers and threads per worker
    const int kMaxNumThreads = -1;
    const int num_eff_threads = GetEffectiveNumThreads(kMaxNumThreads);
    const int kDefaultNumWorkers = 8;
    const int num_eff_workers = 
        options_.num_workers < 1
        ? std::min(static_cast<int>(inter_clusters.size()),
                   std::min(kDefaultNumWorkers, num_eff_threads))
        : options_.num_workers;
    const int num_threads_per_worker = 
        std::max(1, num_eff_threads / num_eff_workers);
    
    // Function to reconstruct one cluster using incremental mapping.
    // TODO: using different kind of mappers to reconstruct, such as global, hybrid
    auto ReconstructCluster = [&, this](
                            const ImageCluster& cluster,
                            ReconstructionManager* reconstruction_manager) {

        IncrementalMapperOptions custom_options = mapper_options_;
        custom_options.max_model_overlap = 3;
        custom_options.init_num_trials = options_.init_num_trials;
        custom_options.num_threads = num_threads_per_worker;

        for (const auto image_id : cluster.image_ids) {
            custom_options.image_names.insert(image_id_to_name.at(image_id));
        }

        IncrementalMapperController mapper(&custom_options, options_.image_path,
                                   options_.database_path,
                                   reconstruction_manager);
        mapper.Start();
        mapper.Wait();
    };

    // Start reconstructing the bigger clusters first for resource usage.
    const auto cmp = [](const ImageCluster& cluster1, const ImageCluster& cluster2) {
                  return cluster1.image_ids.size() > cluster2.image_ids.size(); };
    std::sort(inter_clusters.begin(), inter_clusters.end(), cmp);

    // Start the reconstruction workers.
    reconstruction_managers.reserve(inter_clusters.size());

    bool is_recons_exist = IsPartialReconsExist(reconstructions);
    if (is_recons_exist) {
        LOG(INFO) << "Loaded from previous reconstruction partitions.";
    } else {
        ThreadPool thread_pool(num_eff_workers);
        for (const auto& cluster : inter_clusters) {
            thread_pool.AddTask(ReconstructCluster, cluster, &reconstruction_managers[&cluster]);
        }
        thread_pool.Wait();

        for (const auto& cluster : inter_clusters) {
            auto& recon_manager = reconstruction_managers.at(&cluster);
            for (size_t i = 0; i < recon_manager.Size(); i++) {
                reconstructions.push_back(&recon_manager.Get(i));
            }
        }

        // Export un-transformed partial reconstructions for debugging.
        for (size_t i = 0; i < reconstructions.size(); ++i) {
            const std::string reconstruction_path = JoinPaths(options_.output_path, 
                                                              "partition_" + std::to_string(i));
            CreateDirIfNotExists(reconstruction_path);
            reconstructions[i]->Write(reconstruction_path);
        }
    }
}

void DistributedMapperController::MergeClusters(
    const std::vector<ImageCluster>& inter_clusters,
    std::vector<Reconstruction*>& reconstructions,
    std::unordered_map<const ImageCluster*, ReconstructionManager>& reconstruction_managers,
    const int num_eff_threads)
{
    LOG(INFO) << "Sub-reconstructions size: " << reconstructions.size();

    BundleAdjustmentOptions ba_options = this->GlobalBundleAdjustment();
    ba_options.solver_options.num_threads = num_eff_threads;

    SfMAligner sfm_aligner(reconstructions, ba_options);
    Node anchor_node;
    if (sfm_aligner.Align()) {
        anchor_node = sfm_aligner.GetAnchorNode();
    }
    CHECK_NE(anchor_node.id, -1);
    CHECK_NOTNULL(reconstructions[anchor_node.id]);

    LOG(INFO) << "Adding the final cluster...";
    LOG(INFO) << "Registered images number: " 
              << reconstructions[anchor_node.id]->RegImageIds().size();
    LOG(INFO) << "Reconstructed 3D points: "
              << reconstructions[anchor_node.id]->NumPoints3D();

    // Reading un-transformed reconstruction partitions.
    std::vector<Reconstruction*> trans_recons;
    CHECK_EQ(IsPartialReconsExist(trans_recons), true);

    std::vector<Sim3> sim3_to_anchor = sfm_aligner.GetSim3ToAnchor();
    for (uint i = 0; i < reconstructions.size(); i++) {
        if (i == anchor_node.id) continue;

        Sim3 sim3 = sim3_to_anchor[i];
        Eigen::Vector4d qvec;
        ceres::RotationMatrixToQuaternion(sim3.R.data(), qvec.data());
        SimilarityTransform3 tform(sim3.s, qvec, sim3.t);
        trans_recons[i]->Transform(tform);
    }

    // Insert a new reconstruction manager for merged cluster.
    const ImageCluster root_cluster = image_clustering_->GetRootCluster();
    auto& reconstruction_manager = reconstruction_managers[&root_cluster];
    reconstruction_manager.Add();
    reconstruction_manager.Get(reconstruction_manager.Size() - 1) = 
        *reconstructions[anchor_node.id];

    LOG(INFO) << "Erasing clusters...";
    for (const ImageCluster& inter_cluster : inter_clusters) {
        reconstruction_managers.erase(&inter_cluster); 
    }
    CHECK_EQ(reconstruction_managers.size(), 1);
    *reconstruction_manager_ = std::move(reconstruction_managers.begin()->second);
}

} // namespace GraphSfM