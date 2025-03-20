import json
import subprocess
import packaging.specifiers
import packaging.version

def get_pipdeptree_data():
    """Extracts dependency tree in JSON format."""
    result = subprocess.run(["pipdeptree", "--json"], capture_output=True, text=True)
    return json.loads(result.stdout)

def get_version_constraints(package_name, dependency_data):
    """Finds and merges version constraints for a package."""
    constraints = set()

    for package in dependency_data:
        if package["package"]["key"] == package_name:
            # Only add installed version if it exists
            if "version" in package["package"]:
                constraints.add(f"=={package['package']['version']}")

            # Add constraints from dependencies
            for dep in package["dependencies"]:
                if dep.get("required_version"):
                    constraints.add(dep["required_version"])

    # Merge constraints
    specifier_set = packaging.specifiers.SpecifierSet()
    for constraint in constraints:
        try:
            specifier_set &= packaging.specifiers.SpecifierSet(constraint)
        except packaging.specifiers.InvalidSpecifier:
            continue  # Ignore invalid specifiers

    return str(specifier_set) if specifier_set else ""

def generate_requirements_file(output_file="requirements_custom.txt"):
    """Creates a requirements.txt file with merged package constraints."""
    dependency_data = get_pipdeptree_data()
    package_constraints = {}

    for package in dependency_data:
        package_name = package["package"]["key"]
        constraints = get_version_constraints(package_name, dependency_data)

        if constraints:
            package_constraints[package_name] = f"{package_name}{constraints}"
        else:
            package_constraints[package_name] = package_name  # No constraints

    # Save to requirements.txt
    with open(output_file, "w") as f:
        for package, constraint in sorted(package_constraints.items()):
            f.write(f"{constraint}\n")

    print(f"Requirements file '{output_file}' created successfully.")

# Run the script
generate_requirements_file()
