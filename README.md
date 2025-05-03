# PDD-JonathanGonzalez
 Counting Tree Application

## Considerations

As of April 22nd 2025, the lastest version of Ultralytics has a bug preventing model from passing the imgsz parameter correctly when predicting as explained in [this](https://github.com/ultralytics/ultralytics/issues/20287#issue-3011824471) issue raised on their github page. Due to this a prerelease version of ultralytics is being use. Users should run the following command:

`
pip install git+https://github.com/ultralytics/ultralytics.git@main
`