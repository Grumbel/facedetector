// Facedetector based on dlib
// Copyright (C) 2018 Ingo Ruhnke <grumbel@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <fmt/format.h>
#include <queue>
#include <mutex>
#include <algorithm>
#include <memory>
#include <thread>
#include <future>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>
#include <sstream>

using namespace std;

void process_image(std::string arg, int i, dlib::shape_predictor& sp)
{
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  std::cout << "processing image " << arg << std::endl;

  dlib::array2d<dlib::rgb_pixel> img;
  dlib::load_image(img, arg);
  //dlib::pyramid_up(img);
  auto dets = detector(img);
  std::cout << "Number of faces detected: " << dets.size() << std::endl;

  // Now we will go ask the shape_predictor to tell us the pose of
  // each face we detected.
  std::vector<dlib::full_object_detection> shapes;
  for (unsigned long j = 0; j < dets.size(); ++j)
  {
    dlib::full_object_detection shape = sp(img, dets[j]);
    shapes.push_back(shape);
  }

  float angle = 0.0f;
  for (auto const& shape: shapes)
  {
    for (size_t k = 0; k < shape.num_parts(); ++k)
    {
      auto pnt = shape.part(k);
      dlib::fill_rect(img,
                      dlib::rectangle(pnt.x() - 2, pnt.y() - 2,
                                      pnt.x() + 2, pnt.y() + 2),
                      dlib::rgb_pixel(255, 255, k));
    }

    auto left_eye = dlib::point((shape.part(37) +
                                 shape.part(38) +
                                 shape.part(41) +
                                 shape.part(40)) * 0.25);

    auto right_eye = dlib::point((shape.part(43) +
                                  shape.part(44) +
                                  shape.part(47) +
                                  shape.part(46)) * 0.25);

    auto angle_line = (right_eye - left_eye);
    angle = atan2(angle_line.y(), angle_line.x());
  }

  // dlib::save_jpeg(img, "/tmp/test.jpg");
  //dlib::array<dlib::array2d<dlib::rgb_pixel> > chips;
  //dlib::extract_image_chips(img, dlib::get_face_chip_details(shapes), chips, 512);

  std::vector<dlib::chip_details> det_chip(dets.begin(), dets.end());
  dlib::array<dlib::array2d<dlib::rgb_pixel> > chips;
  dlib::extract_image_chips(img, det_chip, chips);

  for (auto const& chip: chips)
  {
    dlib::array2d<dlib::rgb_pixel> out;
    auto transform = dlib::rotate_image(chip, out, angle);

    std::string outfile = fmt::format("/tmp/out/chip{:08}.jpg", i);
    dlib::save_jpeg(out, outfile);
  }

}

int main(int argc, char** argv)
{
  try
  {
    if (argc == 1)
    {
      std::cout << "Call this program like this:" << std::endl;
      std::cout << argv[0] << " shape_predictor_68_face_landmarks.dat faces/*.jpg" << std::endl;
      std::cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
      std::cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << std::endl;
      return 0;
    }

    std::mutex filenames_mutex;
    std::queue<std::string> filenames;

    for(int i = 2; i < argc; ++i)
    {
      filenames.emplace(argv[i]);
    }

    std::cout << "launching threads" << std::endl;
    std::vector<std::thread> threads;
    for(unsigned int i = 0; i < std::thread::hardware_concurrency(); ++i)
    {
      std::thread thread(
        [&argv, &filenames, &filenames_mutex]
        {
          dlib::shape_predictor sp;
          dlib::deserialize(argv[1]) >> sp;

          while(true)
          {
            int count = 0;
            std::string filename;
            {
              std::lock_guard<std::mutex> lock(filenames_mutex);
              if (filenames.empty())
              {
                return;
              }
              else
              {
                count = filenames.size();
                filename = filenames.front();
                filenames.pop();
              }
            }

            process_image(filename, count, sp);
          }
        });

      threads.push_back(std::move(thread));
    }

    std::cout << "waiting for threads" << std::endl;
    for(auto& thread: threads)
    {
      thread.join();
    }
  }
  catch (exception const& e)
  {
    std::cout << "\nexception thrown!" << std::endl;
    std::cout << e.what() << std::endl;
  }
}


/* EOF */
