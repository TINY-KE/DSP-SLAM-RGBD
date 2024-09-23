#include "include/core/PriorInfer.h"
#include "src/config/Config.h"

using namespace EllipsoidSLAM;
using namespace std;

int main()
{
    EllipsoidSLAM::Config::Init();  // 这个设计很烂，不加这个居然会报错!

    PriFactor pf;

    const std::string str_path = "/disk/workspace/datasets/ICL-NUIM/living_room_traj2n_frei_png/pritable.txt";
    pf.LoadPriConfigurations(str_path);

    // test several pri
    Pri pri = pf.CreatePri(75);
    pri.print();

    return 0;
}