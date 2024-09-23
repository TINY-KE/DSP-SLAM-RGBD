#include "include/core/Map.h"
#include "src/symmetry/PointCloudFilter.h"

using namespace std;

namespace EllipsoidSLAM
{

Map::Map() {
    mCameraState = new g2o::SE3Quat();
}

void Map::addPoint(EllipsoidSLAM::PointXYZRGB *pPoint) {
    unique_lock<mutex> lock(mMutexMap);
    mspPoints.insert(pPoint);
}

void Map::addPointCloud(EllipsoidSLAM::PointCloud *pPointCloud) {
    unique_lock<mutex> lock(mMutexMap);

    for(auto iter=pPointCloud->begin();iter!=pPointCloud->end();++iter){
        mspPoints.insert(&(*iter));
    }
}

void Map::clearPointCloud() {
    unique_lock<mutex> lock(mMutexMap);

    mspPoints.clear();
}

void Map::addEllipsoid(ellipsoid *pObj)
{
    unique_lock<mutex> lock(mMutexMap);
    mspEllipsoids.push_back(pObj);
}


vector<ellipsoid*> Map::GetAllEllipsoids()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspEllipsoids;
}

std::vector<PointXYZRGB*> Map::GetAllPoints() {
    unique_lock<mutex> lock(mMutexMap);
    return vector<PointXYZRGB*>(mspPoints.begin(),mspPoints.end());
}

void Map::addPlane(plane *pPlane, int visual_group)
{
    unique_lock<mutex> lock(mMutexMap);
    pPlane->miVisualGroup = visual_group;
    mspPlanes.insert(pPlane);
}


vector<plane*> Map::GetAllPlanes()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<plane*>(mspPlanes.begin(),mspPlanes.end());
}

void Map::setCameraState(g2o::SE3Quat* state) {
    unique_lock<mutex> lock(mMutexMap);
    mCameraState =  state;
}

void Map::addCameraStateToTrajectory(g2o::SE3Quat* state) {
    unique_lock<mutex> lock(mMutexMap);
    mvCameraStates.push_back(state);
}

void Map::ClearCameraTrajectory() {
    unique_lock<mutex> lock(mMutexMap);
    mvCameraStates.clear();
}

g2o::SE3Quat* Map::getCameraState() {
    unique_lock<mutex> lock(mMutexMap);
    return mCameraState;
}

std::vector<g2o::SE3Quat*> Map::getCameraStateTrajectory() {
    unique_lock<mutex> lock(mMutexMap);
    return mvCameraStates;
}

std::vector<ellipsoid*> Map::getEllipsoidsUsingLabel(int label) {
    unique_lock<mutex> lock(mMutexMap);

    std::vector<ellipsoid*> mvpObjects;
    auto iter = mspEllipsoids.begin();
    for(; iter!=mspEllipsoids.end(); iter++)
    {

        if( (*iter)->miLabel == label )
            mvpObjects.push_back(*iter);

    }

    return mvpObjects;
}

std::map<int, ellipsoid*> Map::GetAllEllipsoidsMap() {
    std::map<int, ellipsoid*> maps;
    for(auto iter= mspEllipsoids.begin(); iter!=mspEllipsoids.end();iter++)
    {
        maps.insert(make_pair((*iter)->miInstanceID, *iter));
    }
    return maps;
}

void Map::clearPlanes(){
    unique_lock<mutex> lock(mMutexMap);
    mspPlanes.clear();
}

void Map::addEllipsoidVisual(ellipsoid *pObj)
{
    unique_lock<mutex> lock(mMutexMap);
    mspEllipsoidsVisual.push_back(pObj);
}


vector<ellipsoid*> Map::GetAllEllipsoidsVisual()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspEllipsoidsVisual;
}

void Map::ClearEllipsoidsVisual()
{
    unique_lock<mutex> lock(mMutexMap);
    mspEllipsoidsVisual.clear();
}


void Map::addEllipsoidObservation(ellipsoid *pObj)
{
    unique_lock<mutex> lock(mMutexMap);
    mspEllipsoidsObservation.push_back(pObj);
}


vector<ellipsoid*> Map::GetObservationEllipsoids()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspEllipsoidsObservation;
}

void Map::ClearEllipsoidsObservation()
{
    unique_lock<mutex> lock(mMutexMap);
    mspEllipsoidsObservation.clear();
}

bool Map::AddPointCloudList(const string& name, std::vector<pcl::PointCloud<pcl::PointXYZRGB>>& vCloudPCL, g2o::SE3Quat& Twc, int type)
{
    if(type == REPLACE_POINT_CLOUD){
        DeletePointCloudList(name, COMPLETE_MATCHING);
    }
    srand(time(0));
    for( auto& cloud : vCloudPCL )
    {
        EllipsoidSLAM::PointCloud cloudQuadri = pclToQuadricPointCloud(cloud);
        EllipsoidSLAM::PointCloud* pCloudGlobal = new EllipsoidSLAM::PointCloud(cloudQuadri);
        transformPointCloudSelf(pCloudGlobal, &Twc);
        
        int r = rand()%155;
        int g = rand()%155;
        int b = rand()%155;
        SetPointCloudProperty(pCloudGlobal, r, g, b, 4);
        bool result = AddPointCloudList(name, pCloudGlobal, ADD_POINT_CLOUD);
        if(!result) {
            delete pCloudGlobal;
            pCloudGlobal = NULL;
        }
    }
    return true;
}

bool Map::AddPointCloudList(const string& name, PointCloud* pCloud, int type){
    unique_lock<mutex> lock(mMutexMap);
    if(pCloud == NULL)
    {
        std::cout << "NULL point cloud." << std::endl;
        return false;
    }

    // Check repetition
    if(mmPointCloudLists.find(name) != mmPointCloudLists.end() )
    {
        // Exist
        auto pCloudInMap = mmPointCloudLists[name];
        if(pCloudInMap==NULL){
            std::cout << "Error: the cloud " << name << " has been deleted." << std::endl;
            return false;
        }

        if( type == 0){
            // replace it.
            pCloudInMap->clear(); // release it
            mmPointCloudLists[name] = pCloud;
        }
        else if( type == 1 )
        {
            // add together
            for( auto &p : *pCloud )
                pCloudInMap->push_back(p);
        }
        else 
        {
            std::cout << "Wrong type : " << type << std::endl;
        }

        return false;
    }
    else{
        mmPointCloudLists.insert(make_pair(name, pCloud));
        return true;
    }
        
}

bool Map::DeletePointCloudList(const string& name, int type){
    unique_lock<mutex> lock(mMutexMap);

    if( type == 0 ) // complete matching: the name must be the same
    {
        auto iter = mmPointCloudLists.find(name);
        if (iter != mmPointCloudLists.end() )
        {
            PointCloud* pCloud = iter->second;
            if(pCloud!=NULL)
            {
                delete pCloud;
                pCloud = NULL;
            }
            mmPointCloudLists.erase(iter);
            return true;
        }
        else{
            std::cerr << "PointCloud name " << name << " doesn't exsit. Can't delete it." << std::endl;
            return false;
        }
    }
    else if ( type == 1 ) // partial matching
    {
        bool deleteSome = false;
        for( auto iter = mmPointCloudLists.begin();iter!=mmPointCloudLists.end();)
        {
            auto strPoints = iter->first;
            if( strPoints.find(name) != strPoints.npos )
            {
                PointCloud* pCloud = iter->second;
                if(pCloud!=NULL)
                {
                    delete pCloud;
                    pCloud = NULL;
                }
                iter = mmPointCloudLists.erase(iter);
                deleteSome = true;
                continue;
            }
            iter++;
        }
        return deleteSome;
    }
    
    return false;
}

bool Map::ClearPointCloudLists(){
    unique_lock<mutex> lock(mMutexMap);

    mmPointCloudLists.clear();
    return true;
}

std::map<string, PointCloud*> Map::GetPointCloudList(){
    unique_lock<mutex> lock(mMutexMap);
    return mmPointCloudLists;
}

PointCloud Map::GetPointCloudInList(const string& name){
    unique_lock<mutex> lock(mMutexMap);

    if(mmPointCloudLists.find(name)!=mmPointCloudLists.end())
        return *mmPointCloudLists[name];
    else
        return PointCloud();    // 空
}

void Map::addBoundingbox(Boundingbox* pBox)
{
    unique_lock<mutex> lock(mMutexMap);
    mvBoundingboxes.push_back(pBox);
}

std::vector<Boundingbox*> Map::GetBoundingboxes()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvBoundingboxes;
}

void Map::ClearBoundingboxes()
{
    unique_lock<mutex> lock(mMutexMap);
    mvBoundingboxes.clear();
}

void Map::addToTrajectoryWithName(SE3QuatWithStamp* state, const string& name)
{
    unique_lock<mutex> lock(mMutexMap);

    // 没有则生成一个
    if( mmNameToTrajectory.find(name) == mmNameToTrajectory.end() ){
        mmNameToTrajectory[name] = Trajectory();
        mmNameToTrajectory[name].push_back(state);
    }
    else
    {
        mmNameToTrajectory[name].push_back(state);
    }
}

Trajectory Map::getTrajectoryWithName(const string& name)
{
    unique_lock<mutex> lock(mMutexMap);
    if( mmNameToTrajectory.find(name) == mmNameToTrajectory.end() ){
        return Trajectory();
    }
    else
    {
        return mmNameToTrajectory[name];
    }
}

bool Map::clearTrajectoryWithName(const string& name)
{
    unique_lock<mutex> lock(mMutexMap);
    if( mmNameToTrajectory.find(name) != mmNameToTrajectory.end() )
    {
        mmNameToTrajectory[name].clear();
        return true;
    }

    return false;

}

bool Map::addOneTrajectory(Trajectory& traj, const string& name)
{
    unique_lock<mutex> lock(mMutexMap);
    mmNameToTrajectory[name] = traj;

    return true;
}

void Map::addArrow(const Vector3d& center, const Vector3d& norm, const Vector3d& color)
{
    unique_lock<mutex> lock(mMutexMap);
    Arrow ar;
    ar.center = center;
    ar.norm = norm;
    ar.color = color;
    mvArrows.push_back(ar);
    
    return;
}

std::vector<Arrow> Map::GetArrows()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvArrows;
}

void Map::clearArrows()
{
    unique_lock<mutex> lock(mMutexMap);
    mvArrows.clear();
    return;
}


} // namespace 